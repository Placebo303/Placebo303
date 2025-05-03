# ✅ config_final.py -- 配套新版train_final.py使用

from easydict import EasyDict as edict
import os
import yaml

# 创建配置对象
config = edict()
config.img_setting = edict()
config.preprocess = edict()
config.net_setting = edict()
config.Pretrain = edict()
config.TRAIN = edict()
config.Loss = edict()
config.VALID = edict()

# ------------------------------标签名----------------------------------
label = '[Mito]_view3_L5c126b8_trytrytry'
config.label = label

# ------------------------------图像设置----------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

config.img_setting.img_size = 160
config.img_setting.sr_factor = 1
config.img_setting.ReScale_factor = [3, 3]
config.img_setting.Nnum = 3
config.img_setting.n_channels = 8
config.channels_interp = 126
config.n_interp = 1
config.sub_pixel = 3
config.img_setting.n_slices = 41

npy_root = os.path.join(base_dir, 'data_npy', 'Mito_View3_[LF160_SF160_WF480]')
config.img_setting.Target3D = os.path.join(npy_root, 'Stack_normalized/')
config.img_setting.Synth_view = os.path.join(npy_root, 'HR/')
config.img_setting.LFP = os.path.join(npy_root, 'LR/')

# ------------------------------预处理设置----------------------------------
config.preprocess.normalize_mode = 'percentile'
config.preprocess.LFP_type = None
config.preprocess.SynView_type = None
config.preprocess.discard_view = []

# ------------------------------网络结构设置----------------------------------
config.net_setting.SR_model = 'F_Denoise'
config.net_setting.Recon_model = 'F_Recon'
config.net_setting.is_bias = False

# ------------------------------预训练模型设置----------------------------------
config.Pretrain.loading_pretrain_model = False
config.Pretrain.ckpt_dir = '/'

# ------------------------------训练参数----------------------------------
config.TRAIN.test_saving_path = os.path.join(base_dir, 'sample', 'test', config.label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = os.path.join(base_dir, 'checkpoint', config.label)
config.TRAIN.log_dir = os.path.join(base_dir, 'log', config.label)
config.TRAIN.valid_on_the_fly = False

config.TRAIN.sample_ratio = 1.0
config.TRAIN.shuffle_all_data = False
config.TRAIN.shuffle_for_epoch = True
config.TRAIN.device = 0
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 5e-4
config.TRAIN.beta1 = 0.9
config.TRAIN.n_epoch = 150
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 50

# ------------------------------损失函数设置----------------------------------
config.Loss.Ratio = [0.1, 0.9]
config.Loss.SR_loss = {
    'weighted_mae_loss': 1.0,
}
config.Loss.Recon_loss = {
    'weighted_mse_loss': 1.0,
    'edge_loss': 0.1,
}

# ------------------------------验证设置----------------------------------
validation_path = os.path.join(npy_root, 'LR') + '/'
config.VALID.lf2d_path = validation_path
config.VALID.save_type = 'LFP'
config.VALID.saving_path = '{}SR_{}/'.format(config.VALID.lf2d_path, label)
config.VALID.gt_path = config.VALID.gt_path = os.path.join(base_dir, 'data', 'Mito_View3_[LF160_SF160_WF480]', 'Stack')

# ------------------------------YAML加载接口----------------------------------
def _load_yaml_to_edict(d, yaml_data):
    for key, value in yaml_data.items():
        if isinstance(value, dict):
            if key not in d:
                d[key] = edict()
            _load_yaml_to_edict(d[key], value)
        else:
            d[key] = value

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    _load_yaml_to_edict(config, yaml_data)
    config.TRAIN.test_saving_path = os.path.join(base_dir, 'sample', 'test', config.label)
    config.TRAIN.ckpt_dir = os.path.join(base_dir, 'checkpoint', config.label)
    config.TRAIN.log_dir = os.path.join(base_dir, 'log', config.label)
    config.VALID.saving_path = '{}SR_{}/'.format(config.VALID.lf2d_path, config.label)

# 注册
config.load = load_config_from_yaml
