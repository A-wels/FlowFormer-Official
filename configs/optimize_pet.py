from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'default'
_CN.suffix ='pet'
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 6
_CN.sum_freq = 100
_CN.val_freq = 5000000
_CN.image_size = [344, 127]
_CN.add_noise = False
_CN.critical_params = []

_CN.transformer = 'latentcostformer'
_CN.restore_ckpt = 'checkpoints/kitti.pth'

# latentcostformer
_CN.latentcostformer = CN()
_CN.latentcostformer.pe = 'linear'
_CN.latentcostformer.dropout = 0.0
_CN.latentcostformer.encoder_latent_dim = 256 # in twins, this is 256
_CN.latentcostformer.query_latent_dim = 64
_CN.latentcostformer.cost_latent_input_dim = 64
_CN.latentcostformer.cost_latent_token_num = 8
_CN.latentcostformer.cost_latent_dim = 128
_CN.latentcostformer.arc_type = 'transformer'
_CN.latentcostformer.cost_heads_num = 1
# encoder
_CN.latentcostformer.pretrain = True
_CN.latentcostformer.context_concat = False
_CN.latentcostformer.encoder_depth = 3
_CN.latentcostformer.feat_cross_attn = False
_CN.latentcostformer.patch_size = 8
_CN.latentcostformer.patch_embed = 'single'
_CN.latentcostformer.no_pe = False
_CN.latentcostformer.gma = "GMA"
_CN.latentcostformer.kernel_size = 9
_CN.latentcostformer.rm_res = True
_CN.latentcostformer.vert_c_dim = 64
_CN.latentcostformer.cost_encoder_res = True
_CN.latentcostformer.cnet = 'twins'
_CN.latentcostformer.fnet = 'twins'
_CN.latentcostformer.no_sc = False
_CN.latentcostformer.only_global = False
_CN.latentcostformer.add_flow_token = True
_CN.latentcostformer.use_mlp = False
_CN.latentcostformer.vertical_conv = False

# decoder
_CN.latentcostformer.decoder_depth = 12
_CN.latentcostformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-5
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 300
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()

def update_cfg(cfg, settings: dict):
    cfg.gamma = settings['gamma']
    cfg.max_flow = settings['max_flow']
    if settings['ckpt'] == "None":
        cfg.restore_ckpt = None
    else:
        cfg.restore_ckpt = settings['ckpt']

    if 'only_global' in settings:
        cfg.latentcostformer.only_global = settings['only_global']
    if 'feat_cross_attn' in settings:
        cfg.latentcostformer.feat_cross_attn = settings['feat_cross_attn']
    if 'use_mlp' in settings:
        cfg.latentcostformer.use_mlp = settings['use_mlp']
    if 'vertical_conv' in settings:
        cfg.latentcostformer.vertical_conv = settings['vertical_conv']

    cfg.trainer.num_steps = settings['num_steps']
    cfg.trainer.canonical_lr = settings['canonical_lr']
    cfg.trainer.adamw_decay = settings['adamw_decay']
    cfg.trainer.epsilon = settings['epsilon']

    return cfg