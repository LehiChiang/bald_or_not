import os
import numpy as np

import torch as t
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from models import resnet18
from tqdm import tqdm
from terminaltables import AsciiTable
from tensorboardX import SummaryWriter

from config import DefaultConfig
from data.dataset import BaldDataset
from utils.pytorchtools import EarlyStopping
from utils.utils import print_separator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(**kwargs):
    # initialization
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    opt = DefaultConfig()
    train_losses, valid_losses, avg_train_losses, avg_valid_losses = [], [], [], []
    writer = SummaryWriter('logs', comment='resnet18')
    criterion = nn.CrossEntropyLoss().to(device)
    config_data = [['Key', 'Value'], ['device', device]]

    # config
    config_generator(config_data, opt)

    # data
    train_data, train_dataloader, val_data, val_dataloader = data_generator(opt)

    # model
    model = model_generator(device, opt)

    # optimizer & lr_scheduler & early_stopping
    optimizer = Adam(model.fc.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=False, path='checkpoints/%s_final_checkpoint.pth' % opt.model)

    print('Starting training on %d images:' % len(train_data))

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if opt.freeze:
        for epoch in range(opt.freeze_epoch):
            print('Epoch {}/{} :'.format(epoch, opt.freeze_epoch+opt.unfreeze_epoch))
            model.train()

            train_loss, loss_meter, correct = 0, 0, 0

            # train epoch
            for batch_i, (data, label, _) in enumerate(tqdm(train_dataloader)):
                input = Variable(data).to(device)
                target = Variable(label).to(device)
                optimizer.zero_grad()
                predict = model(input)
                loss = criterion(predict, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                loss_meter += loss.item()
                logits = t.relu(predict)
                pred = logits.data.max(1)[1]
                correct += pred.eq(target.data).sum()
            train_acc = correct.cpu().detach().numpy() * 1.0 / len(train_dataloader.dataset)
            # ending of train epoch

            # validation epoch
            if epoch % opt.evaluation_interval == 0:
                if t.cuda.device_count() > 1:
                    model = nn.DataParallel(model, device_ids=[0])
                model.eval()
                loss_meter, correct = 0, 0
                with t.no_grad():
                    print('Validating on %d images:' % len(val_data))
                    for inputs, target, _ in tqdm(val_dataloader):
                        inputs = inputs.to(device)
                        target = target.to(device)
                        output = model(inputs)
                        loss = criterion(output, target)
                        loss_meter += loss.item()
                        valid_losses.append(loss.item())
                        logits = t.relu(output)
                        pred = logits.data.max(1)[1]
                        correct += pred.eq(target.data).sum()
                    val_acc = correct.cpu().detach().numpy() * 1.0 / len(val_dataloader.dataset)
            # ending of validation epoch

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print('train_loss: %.3f, train_acc: %.3f' % (train_loss, train_acc))
            print('val_loss: %.3f, val_acc: %.3f' % (valid_loss, val_acc))
            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('train_acc', train_acc, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('val_acc', val_acc, global_step=epoch)

            # clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                opt.unfreeze = False
                break

            if epoch % opt.checkpoint_interval == 0:
                t.save(model.state_dict(), 'checkpoints/' + '%s_ckpt_%d.pth' % (opt.model, epoch))

            print_separator()

            # load the last checkpoint with the best model
            model.load_state_dict(t.load('checkpoints/' + '%s_final_checkpoint.pth' % opt.model))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if opt.unfreeze:
        print('Unfreeze all layers:')
        for param in model.parameters():
            param.requires_grad = True

        optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=opt.lr_decay, patience=3, verbose=True)

        for epoch in range(opt.freeze_epoch, opt.unfreeze_epoch):
            print('Epoch {}/{} :'.format(epoch, opt.unfreeze_epoch+opt.freeze_epoch))
            model.train()

            train_loss, loss_meter, correct = 0, 0, 0

            # train epoch
            for batch_i, (data, label, _) in enumerate(tqdm(train_dataloader)):
                input = Variable(data).to(device)
                target = Variable(label).to(device)
                optimizer.zero_grad()
                predict = model(input)
                loss = criterion(predict, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                loss_meter += loss.item()
                logits = t.relu(predict)
                pred = logits.data.max(1)[1]
                correct += pred.eq(target.data).sum()
            train_acc = correct.cpu().detach().numpy() * 1.0 / len(train_dataloader.dataset)
            # ending of train epoch

            # validation epoch
            if epoch % opt.evaluation_interval == 0:
                if t.cuda.device_count() > 1:
                    model = nn.DataParallel(model, device_ids=[0])
                model.eval()
                loss_meter, correct = 0, 0
                with t.no_grad():
                    print('Validating on %d images:' % len(val_data))
                    for inputs, target, _ in tqdm(val_dataloader):
                        inputs = inputs.to(device)
                        target = target.to(device)
                        output = model(inputs)
                        loss = criterion(output, target)
                        loss_meter += loss.item()
                        valid_losses.append(loss.item())
                        logits = t.relu(output)
                        pred = logits.data.max(1)[1]
                        correct += pred.eq(target.data).sum()
                    val_acc = correct.cpu().detach().numpy() * 1.0 / len(val_dataloader.dataset)
            # ending of validation epoch

            lr_scheduler.step(loss.item())
            writer.add_scalar('learning_rate', lr_scheduler.state_dict()['_last_lr'], global_step=epoch)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print('train_loss: %.3f, train_acc: %.3f' % (train_loss, train_acc))
            print('val_loss: %.3f, val_acc: %.3f' % (valid_loss, val_acc))
            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('train_acc', train_acc, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('val_acc', val_acc, global_step=epoch)

            # clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch % opt.checkpoint_interval == 0:
                t.save(model.state_dict(), 'checkpoints/' + '%s_ckpt_%d.pth' % (opt.model, epoch))

            print_separator()

            # load the last checkpoint with the best model
            model.load_state_dict(t.load('checkpoints/' + '%s_final_checkpoint.pth' % opt.model))


def config_generator(config_data, opt):
    for k, v in opt.configs.items():
        config_data.append([k, v])
    config_table = AsciiTable(config_data)
    print(config_table.table)


def model_generator(device, opt):
    model = resnet18()
    if opt.load_model_path:
        model.load_state_dict(t.load(opt.load_model_path))
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, opt.nums_of_classes)
    if t.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=opt.device_ids)
    model.to(device)
    return model


def data_generator(opt):
    train_data = BaldDataset(opt.train_data_root)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  collate_fn=train_data.customized_collate_fn,
                                  drop_last=False)
    val_data = BaldDataset(opt.val_data_root)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers,
                                collate_fn=val_data.customized_collate_fn,
                                drop_last=False)
    return train_data, train_dataloader, val_data, val_dataloader


if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("model_data", exist_ok=True)
    train()
