import warnings


class DefaultConfig:
    def __init__(self):
        self._configs = dict()
        self._configs['env'] = None  # visdom环境
        self._configs['model'] = 'resnet18'  # 使用的模型，名字必须与models/__init__.py里的名字一致
        #  'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152'

        # Directories
        self._configs['train_data_root'] = 'data/bald/Train'  # 训练集存放路径
        self._configs['test_data_root'] = 'data/bald/Test'  # 测试集存放路径
        self._configs['val_data_root'] = 'data/bald/Validation'  # 验证集存放路径
        self._configs['load_model_path'] = None  # 'model_data/model.pth'加载预训练模型的路径，None表示不预训练

        # Training Config
        self._configs['batch_size'] = 4
        self._configs['num_workers'] = 0
        self._configs['device_ids'] = [0, 1, 2, 3]
        self._configs['max_epoch'] = 100
        self._configs['lr'] = 0.01
        self._configs['lr_decay'] = 0.95
        self._configs['weight_decay'] = 1e-4
        self._configs['checkpoint_interval'] = 1
        self._configs['evaluation_interval'] = 1

        # Test Config
        self._configs['debug_file'] = 'tmp/debug'
        self._configs['result_file'] = 'output/result.csv'

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: DefaultConfig has no attribute %s' % k)
            setattr(self, k, v)

        print('User Config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, ':', getattr(self, k))

    @property
    def configs(self):
        return self._configs

    @property
    def env(self):
        return self._configs['env']

    @env.setter
    def env(self, value):
        self._configs['env'] = value

    @property
    def model(self):
        return self._configs['model']

    @model.setter
    def model(self, value):
        self._configs['model'] = value

    @property
    def train_data_root(self):
        return self._configs['train_data_root']

    @train_data_root.setter
    def train_data_root(self, value):
        self._configs['train_data_root'] = value

    @property
    def test_data_root(self):
        return self._configs['test_data_root']

    @test_data_root.setter
    def test_data_root(self, value):
        self._configs['test_data_root'] = value

    @property
    def val_data_root(self):
        return self._configs['val_data_root']

    @val_data_root.setter
    def val_data_root(self, value):
        self._configs['val_data_root'] = value

    @property
    def load_model_path(self):
        return self._configs['load_model_path']

    @load_model_path.setter
    def load_model_path(self, value):
        self._configs['load_model_path'] = value

    @property
    def batch_size(self):
        return self._configs['batch_size']    

    @batch_size.setter
    def batch_size(self, value):
        self._configs['batch_size'] = value

    @property
    def use_gpu(self):
        return self._configs['use_gpu']
    
    @use_gpu.setter
    def use_gpu(self, value):
        self._configs['use_gpu'] = value

    @property
    def num_workers(self):
        return self._configs['num_workers']

    @num_workers.setter
    def num_workers(self, value):
        self._configs['num_workers'] = value

    @property
    def device_ids(self):
        return self._configs['device_ids']

    @device_ids.setter
    def device_ids(self, value):
        self._configs['device_ids'] = value

    @property
    def evaluation_interval(self):
        return self._configs['evaluation_interval']

    @evaluation_interval.setter
    def evaluation_interval(self, value):
        self._configs['evaluation_interval'] = value

    @property
    def max_epoch(self):
        return self._configs['max_epoch']

    @max_epoch.setter
    def max_epoch(self, value):
        self._configs['max_epoch'] = value

    @property
    def lr(self):
        return self._configs['lr']

    @lr.setter
    def lr(self, value):
        self._configs['lr'] = value

    @property
    def lr_decay(self):
        return self._configs['lr_decay']

    @lr_decay.setter
    def lr_decay(self, value):
        self._configs['lr_decay'] = value

    @property
    def weight_decay(self):
        return self._configs['weight_decay']

    @weight_decay.setter
    def weight_decay(self, value):
        self._configs['weight_decay'] = value

    @property
    def checkpoint_interval(self):
        return self._configs['checkpoint_interval']

    @checkpoint_interval.setter
    def checkpoint_interval(self, value):
        self._configs['checkpoint_interval'] = value

    @property
    def debug_file(self):
        return self._configs['debug_file']
    
    @debug_file.setter
    def debug_file(self, value):
        self._configs['debug_file'] = value

    @property
    def result_file(self):
        return self._configs['result_file']

    @result_file.setter
    def result_file(self, value):
        self._configs['result_file'] = value
