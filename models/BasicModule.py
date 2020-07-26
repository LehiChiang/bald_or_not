import time

from torch import nn
from torch import load, save

class BasicModule(nn.Module):
    '''
    封装了nn.Module，主要提供save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        '''
        加载指定路径的模型
        :param path:
        :return:
        '''
        self.load_state_dict(load(path))

    def save(self, name=None):
        '''
        保存模型
        :param name:
        :return:
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        save(self.state_dict(), 'checkpoints/' + name)
