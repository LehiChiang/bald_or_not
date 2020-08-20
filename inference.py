import torch as t
import torch.nn as nn
import os
import csv
from torch.utils.data import DataLoader
from models import resnet18

from config import DefaultConfig
from data.dataset import BaldDataset

def test():
    test_config = {
        'model_path': 'checkpoints/resnet18_final_checkpoint.pth',
        'visualize': False
    }
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    opt = DefaultConfig()
    model = resnet18().eval()
    model.fc = nn.Linear(model.fc.in_features, opt.nums_of_classes)
    model.load_state_dict(t.load(os.path.join(os.getcwd(), test_config['model_path'])))
    model.to(device)

    # data
    test_data = BaldDataset(opt.test_data_root)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers,
                                 collate_fn=test_data.customized_collate_fn,
                                 drop_last=False)

    correct = 0
    print('Start inference.....')
    with t.no_grad():
        f = open(opt.result_file, 'w+', encoding='utf-8', newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['image', 'detection_result', 'ground_truth'])
        for inputs, target, img_names in test_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            pred = t.relu(output).data.max(1)[1]
            correct += pred.eq(target.data).sum()
            for img_name, dr, gt in zip(img_names, pred, target.data):
                csv_writer.writerow([img_name, dr.item(), gt.item()])
        test_acc = correct.cpu().detach().numpy() * 1.0 / len(test_dataloader.dataset)
        f.close()
        print("test_acc: %.3f \n" % test_acc)


if __name__ == '__main__':
    test()
