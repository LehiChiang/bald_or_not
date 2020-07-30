import os
import numpy as np

import torch as t
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from terminaltables import AsciiTable

from config import DefaultConfig
from data.dataset import BaldDataset
import models
from utils.pytorchtools import EarlyStopping
from utils.utils import print_separator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def train(**kwargs):

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    opt = DefaultConfig()
    model = getattr(models, opt.model)()
    criterion = nn.CrossEntropyLoss().to(device)
    config_data = [['Key', 'Value'], ['device', device]]

    # print config
    for k, v in opt.configs.items():
        config_data.append([k, v])
    config_table = AsciiTable(config_data)
    print(config_table.table)

    # model
    global model
    if opt.load_model_path:
        model.load_state_dict(t.load(opt.load_model_path))
    if t.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=opt.device_ids)
    model.to(device)

    # data
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
                                shuffle=False,
                                num_workers=opt.num_workers,
                                collate_fn=val_data.customized_collate_fn,
                                drop_last=False)

    # optimizer & lr_scheduler & early_stopping
    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses, valid_losses, avg_train_losses, avg_valid_losses = [], [], [], []

    print('Starting training on %d images:' % len(train_data))

    for epoch in range(opt.max_epoch):

        print('Epoch {}/{} :'.format(epoch, opt.max_epoch))
        model.train()

        train_loss, loss_meter, correct = 0, 0, 0

        # train epoch
        for batch_i, (data, label) in enumerate(tqdm(train_dataloader)):
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
                for inputs, target in tqdm(val_dataloader):
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

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print('train_loss: %.3f, train_acc: %.3f' % (train_loss, train_acc))
        print("val_loss: %.3f, val_acc: %.3f \n" % (valid_loss, val_acc))

        # clear lists to track next epoch
        train_losses.clear()
        valid_losses.clear()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model, path='%s_checkpoint.pth' % opt.model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # if epoch % opt.checkpoint_interval == 0:
        #     model.save('%s_ckpt_%d.pth' % (opt.model, epoch))

        print_separator()

        # load the last checkpoint with the best model
        model.load('%s_checkpoint.pth' % opt.model)


if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("model_data", exist_ok=True)

    train()
