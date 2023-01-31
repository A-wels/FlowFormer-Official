import numpy as np
from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from train_FlowFormer import train_with_parameters

def ff_from_cfg(cfg):
    config_dict = {k: cfg[k] for k in cfg if cfg[k]}
    from configs.optimize_pet import get_cfg, update_cfg

    cfg = get_cfg()
    cfg = update_cfg(cfg, config_dict)
    train_with_parameters(cfg)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    model.cuda()
    model.eval()
    results = evaluate.validate_pet(model.module)
    print(results)
    exit()

if __name__ == "__main__":
    # Build configspace

    cs = ConfigurationSpace(seed=0)

    # set up hyperparameters
    gamma = UniformFloatHyperparameter("gamma", 0.0001, 3, default_value=0.85, log=True)
    max_flow = UniformIntegerHyperparameter("max_flow", 2, 400, default_value=400, log=True)
    ckpt = CategoricalHyperparameter("ckpt", ['checkpoints/kitti.pth', 'checkpoints/sintel.pth', "None"], default_value='checkpoints/sintel.pth')
    
    only_global = CategoricalHyperparameter("only_global", [True, False], default_value=False)
    feat_cross_attn = CategoricalHyperparameter("feat_cross_attn", [True, False], default_value=False)
    use_mlp = CategoricalHyperparameter("use_mlp", [True, False], default_value=False)
    vertical_conv = CategoricalHyperparameter("vertical_conv", [True, False], default_value=False)

    num_steps = UniformIntegerHyperparameter("num_steps", 50, 500, default_value=50, log=True)
    canonical_lr = UniformFloatHyperparameter("canonical_lr", 12.5e-7, 12.5e-3, default_value=12.5e-5, log=True)
    adamw_decay = UniformFloatHyperparameter("adamw_decay", 1e-7, 1e-3, default_value=1e-5, log=True)
    epsilon = UniformFloatHyperparameter("epsilon", 1e-10, 1e-5, default_value=1e-8, log=True)

    cs.add_hyperparameters([gamma, max_flow, ckpt, only_global, feat_cross_attn, use_mlp, vertical_conv, num_steps, canonical_lr,adamw_decay, epsilon])

   # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 50,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": True,
        }
    )
   
    def_value = ff_from_cfg(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))