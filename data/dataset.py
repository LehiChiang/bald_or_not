import os
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms as Trans, ToPILImage
from torchvision.utils import make_grid, save_image


class BaldDataset(data.Dataset):
    '''
    秃头数据集
    '''
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.to_img = ToPILImage()

        bald_url = os.path.join(self.root, 'Bald')
        not_bald_url = os.path.join(self.root, 'NotBald')

        self.image_list = [os.path.join(bald_url, img) for img in os.listdir(bald_url)]
        self.image_list.extend([os.path.join(not_bald_url, img) for img in os.listdir(not_bald_url)])

        if self.transforms is None:
            self.transforms = Trans.Compose([
                Trans.Resize((218, 178)),
                Trans.ToTensor(),
                # Trans.Normalize(mean=[0.485, 0.456, 0.406],
                #                 std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = 0 if img_path.split('/')[-2] == 'Bald' else 1
        img_data = Image.open(img_path)
        img_data = self.transforms(img_data)
        return img_data, label, img_path.split('\\')[-1]

    def __len__(self):
        return len(self.image_list)

    def images_stitching(self, tensor, num_in_row, visualize=True, savefile=False):
        '''
        图片拼接
        :param tensor: 图片Tensor
        :param num_in_row: 每行展示的图片数
        :param visualize: 是否将拼接结果展示
        :param savefile: 是否保存图片
        :return:
        '''
        stitching = make_grid(tensor, num_in_row)
        img = self.to_img(stitching)
        if visualize:
            img.show()
        if savefile:
            save_image(stitching, 'stitching.png')

    def customized_collate_fn(self, batch):
        '''
        数据加载异常时的batch重新分配函数，将加载为None的对象过滤掉，返回可用的数据
        :param batch:
        :return: default_collate
        '''
        batch = list(filter(lambda img: img[0] is not None, batch))
        return default_collate(batch)


if __name__ == '__main__':
    test_data = BaldDataset('bald/Test')
    test_loader = DataLoader(dataset=test_data,
                             batch_size=4,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=test_data.customized_collate_fn,
                             drop_last=False)

    test_data_iter = iter(test_loader)
    img_data, img_label, img_name = next(test_data_iter)
    test_data.images_stitching(img_data, 2)

