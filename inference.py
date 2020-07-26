import torch as t
import os

from torch.utils.data import DataLoader

from config import DefaultConfig
import models
from data.dataset import BaldDataset


def test():
    test_config = {
        'model_path': 'model_data/bald_weights.pth',
        'visualize': False
    }
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    opt = DefaultConfig()
    model = getattr(models, opt.model)()
    model.load(os.path.join(os.getcwd(), test_config['model_path']))
    model.to(device)

    # data
    test_data = BaldDataset(opt.test_data_root)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=opt.batch_size,
                                 shuffle=True,
                                 num_workers=opt.num_workers,
                                 collate_fn=test_data.customized_collate_fn,
                                 drop_last=False)

    model.eval()
    correct = 0
    with t.no_grad():
        for inputs, target in test_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)

            logits = t.relu(output)
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        test_acc = correct.cpu().detach().numpy() * 1.0 / len(test_dataloader.dataset)
        print("test_acc: %.3f \n" % test_acc)


if __name__ == '__main__':
    test()
