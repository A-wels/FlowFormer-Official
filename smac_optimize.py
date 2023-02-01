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
import torch
from core.FlowFormer import build_flowformer
from configs.pet_eval import get_cfg as get_eval_cfg
import evaluate_FlowFormer as evaluate

def ff_from_cfg(cfg):
    config_dict = {k: cfg[k] for k in cfg if cfg[k]}
    from configs.optimize_pet import get_cfg, update_cfg

    cfg = get_cfg()
    cfg = update_cfg(cfg, config_dict)
    train_with_parameters(cfg)

    cfg = get_eval_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    print(cfg.model)
    model.load_state_dict(torch.load(cfg.model))
    model.cuda()
    model.eval()
    results = evaluate.validate_pet(model.module)
    return results['epe']

if __name__ == "__main__":
    # Build configspace

    cs = ConfigurationSpace(seed=0)

    # set up hyperparameters
    gamma = UniformFloatHyperparameter("gamma", 0.0001, 5, default_value=0.85, log=True)
    max_flow = UniformIntegerHyperparameter("max_flow", 5, 10000, default_value=400, log=True)
    ckpt = CategoricalHyperparameter("ckpt", ["None",'checkpoints/sintel.pth','checkpoints/kitti.pth'], default_value='checkpoints/sintel.pth')
    #ckpt = CategoricalHyperparameter("ckpt", ["checkpoints/sintel.pth"], default_value="checkpoints/sintel.pth")
    
    #only_global = CategoricalHyperparameter("only_global", [True], default_value=True)
    #feat_cross_attn = CategoricalHyperparameter("feat_cross_attn", [True], default_value=True)
    context_contact = CategoricalHyperparameter("context_contact", [False], default_value=False)

    num_steps = UniformIntegerHyperparameter("num_steps", 300, 1000, default_value=500, log=True)
    canonical_lr = UniformFloatHyperparameter("canonical_lr", 12.5e-7, 12.5e-3, default_value=12.5e-5, log=True)
    adamw_decay = UniformFloatHyperparameter("adamw_decay", 1e-7, 1e-1, default_value=1e-5, log=True)
    epsilon = UniformFloatHyperparameter("epsilon", 1e-15, 1e-5, default_value=1e-8, log=True)

    cs.add_hyperparameters([gamma, max_flow, ckpt, context_contact, num_steps, canonical_lr,adamw_decay, epsilon])
  
   # cs.add_condition(InCondition(child=only_global, parent=ckpt, values=["None"]))
   # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 100,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": True,
        }
    )
   # def_value: 3.80 after manual testing
  #  def_value = ff_from_cfg(cs.get_default_configuration())
    def_value = 3.80
    print("Default Value: %.2f" % (def_value))
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42), tae_runner=ff_from_cfg)
    incumbent = smac.optimize()
    inc_value = ff_from_cfg(incumbent)
    print("Optimized Value: %.2f" % (inc_value))
    print("----------")
    print(incumbent)
