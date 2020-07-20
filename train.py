import os
import time

import torch as t
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DefaultConfig
from data.dataset import BaldDataset
from models.resnet import resnet50

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
opt = DefaultConfig()
model = resnet50()
criterion = CrossEntropyLoss()

def train(**kwargs):
    # model
    global model
    if opt.load_model_path:
        model.load_state_dict(t.load(opt.load_model_path))
    if opt.use_gpu: model.cuda()

    # data
    train_data = BaldDataset(opt.train_data_root)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  collate_fn=train_data.customized_collate_fn,
                                  drop_last=False)

    #Loss function & optimizer
    lr = opt.lr
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    #train
    for epoch in range(opt.max_epoch):
        print('Epoch {}/{} :'.format(epoch, opt.max_epoch))
        model.train()
        start_time = time.time()
        loader = tqdm(train_dataloader)
        train_loss, loss_meter, it_count, correct = 0, 0, 0, 0
        # train epoch
        for batch_i, (data, label) in enumerate(loader):
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input.cuda()
                target.cuda()
            optimizer.zero_grad()
            predict = model(input)
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()

            loader.set_description("Training")

            loss_meter += loss.item()
            it_count += 1

            logits = t.sigmoid(predict)
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()
        # ending of train epoch

        time_elapsed = time.time() - start_time
        train_loss = loss_meter / len(train_dataloader.dataset)
        train_acc = correct.cpu().detach().numpy() * 1.0 / len(train_dataloader.dataset)

        print('#epoch: %02d, train_loss: %.3e, train_acc: %.3f time: %dm%ds'
              % (epoch, train_loss, train_acc, time_elapsed // 60, time_elapsed % 60))

        # validation
        if epoch % opt.evaluation_interval == 0:
            val_loss, val_acc = val()

        if epoch % opt.checkpoint_interval == 0:
            model.save(f"checkpoints/%s_ckpt_%d.pth" % model.model_name, epoch)


def val(args=None):
    #data
    val_data = BaldDataset(opt.val_data_root)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                collate_fn=val_data.customized_collate_fn,
                                drop_last=False)
    global model
    if args is not None and args.ckpt is not None:
        model.load_state_dict(t.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)

    model.eval()
    loss_meter, correct = 0, 0
    with t.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()

            logits = t.sigmoid(output)
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        val_loss = loss_meter / len(val_data)
        val_acc = correct.cpu().detach().numpy() * 1.0 / len(val_dataloader.dataset)
        print("Val loss: %.3e, Val Acc: %.3f \n" % (val_loss, val_acc))

    return val_loss, val_acc


if __name__ == '__main__':
    # os.makedirs("output", exist_ok=True)
    # os.makedirs("checkpoints", exist_ok=True)
    train()
    # bar = tqdm(["a", "b", "c", "d"])
    # for char in bar:
    #     bar.set_description("Processing %s" % char)
    #     time.sleep(1)
