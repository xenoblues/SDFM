import yaml
import os
import time
from utils import util, torch, generate_pad


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index


def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        setattr(cfg, k, v)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype

    cfg.dct_m, cfg.idct_m = util.get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    # index = get_log_dir_index(cfg.base_dir)
    formatted_time = "_" + time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    if args_dict['mode'] == 'train' or  args_dict['mode'] == 'pred' or args_dict['mode'] == 'eval':
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['cfg'] + '_' +args_dict['mode'] + formatted_time)
    else:
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['mode'] + formatted_time)

    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)

    # flow matching
    cfg.ode_method = args_dict['ode_method']
    cfg.ode_options = args_dict['ode_options']
    # cfg.cfg_scale = args_dict['cfg_scale']
    cfg.skewed_timesteps = args_dict['skewed_timesteps']
    cfg.edm_schedule = args_dict['edm_schedule']
    cfg.generator = args_dict['generator']
    cfg.model_name = args_dict['model_name']
    if cfg.save_metrics_interval == -1:
        cfg.save_metrics_interval = cfg.num_epoch - 1
    return cfg


class Config:
    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'inference' if test else 'results'
        os.makedirs(self.base_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg['batch_size']
        self.normalize_data = cfg.get('normalize_data', False)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']

        self.num_epoch = cfg['num_epoch']
        self.num_data_sample = cfg['num_data_sample']
        self.num_val_data_sample = cfg['num_val_data_sample']
        self.lr = cfg['lr']

        self.n_pre = cfg['n_pre']
        self.multimodal_path = cfg['multimodal_path']
        self.data_candi_path = cfg['data_candi_path']

        self.padding = cfg['padding']
        self.num_layers = cfg['num_layers']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']

        self.mod_train = cfg['mod_train']
        self.mod_test = cfg['mod_test']

        self.use_dct = cfg['use_dct']
        self.dct_norm_enable = cfg['dct_norm_enable']

        self.resume = cfg['resume']
        self.ckpt_path = cfg['ckpt_path']
        self.frame_mask = cfg['frame_mask']
        self.joint_mask = cfg['joint_mask']
        self.residual_data = cfg['residual_data']
        self.random_sample = cfg['random_sample']

        self.cross_attention = cfg['cross_attention']
        self.stylization_block = cfg['stylization_block']
        self.flash_attention = cfg['flash_attention']
        self.se_layer = cfg['se_layer']
        self.milestone = cfg['milestone']
        self.cfg_scale = cfg['cfg_scale']
        self.skip_type = cfg['skip_type']


        # indirect variable
        self.joint_num = 16 if self.dataset == 'h36m' else 14
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
        self.t_total = self.t_his + self.t_pred
