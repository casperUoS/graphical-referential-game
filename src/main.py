import sys

from data import *
from train import *
from eval import *
import os

import argparse

##### INI ######################################################################
from utils import device, path

##### CONFIG
config = {
    "exp_name": "test",  # Experiment name
    "seed": 0,  # seed
    "nb_agents": 2,  # Number of agents
    "agents_path": None,  # Exp. Path of pre-trained agents (None if no pre-training)
    "n_features": 5,
    "embedding_size": 32,  # Embedding size of encoders
    "assoc_lr": 0.0001,  # LR for referent-utterance associations
    "action_lr": 0.01,  # LR for action generation
    "action_bs": 64,  # Nb of simultaneous search during action generation
    "action_it": 100,  # Nb of action generation iterations
    "action_size": 40,  # Size of agent's actions vectors
    "referents_bs": 32,  # Nb of games per training iteration
    "max_iterations": 10000,  # Max nb of training iterations
    "use_img_perspectives": True,  # Convert vectors referents into MNIST compositions?
    "ss_class": "dmp",  # Sensorimotor system class
    "ss_params": {"n_bfs": 20, "dt": 1e-1, "n": 10, "d": 52, "th": 1e-2},  # Sensorimotor system params
    "shared_perspective": False,
    "ood": False,
    "nb_features": 5,
    "bins": [1],
    "transfer_refs": None,
    "use_temp": True,
    "use_baseline": True,
    "no_ss": False
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iterations", type=int, default=100000)

    params = sys.argv[1:]
    opts = parser.parse_args(params)

    return opts

################################################################################

if __name__ == "__main__":

    # if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
    #     seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # else:
    #     seed = 0
    #
    # if "SLURM_JOB_NAME" in os.environ.keys():
    #     exp_name = os.environ["SLURM_JOB_NAME"]
    # else:
    #     exp_name = "one-hot-no-ss"

    opts = parse_arguments()

    exp_name = opts.exp_name
    seed = opts.seed

    if exp_name == "test":
        # !!!! CHANGE BACK TO 100/10000
        config["action_it"] = 10
        config["max_iterations"] = 10

        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
    elif exp_name == "base":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
    elif exp_name == "base-shared":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
        config["shared_perspective"] = True
    elif exp_name == "base-compo":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
        config["bins"] = [1, 2]
    elif exp_name == "base-compo-noTemp":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
        config["bins"] = [1, 2]
        config["use_temp"] = False
    elif exp_name == "base-compo-noBaseline":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
        config["bins"] = [1, 2]
        config["use_baseline"] = False
    elif exp_name == "base-transfer":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
        config["bins"] = [1, 2]
        config["transfer_refs"] = torch.tensor([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
    elif exp_name == "n-10":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 10
        config["nb_features"] = 5
        config["ood"] = False
    elif exp_name == "n-100":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 100
        config["nb_features"] = 5
        config["ood"] = False
    elif exp_name == "f-10":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 10
        config["ood"] = False
    elif exp_name == "ood":
        config["max_iterations"] = 5000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = True
    elif exp_name == "one-hot":
        config["max_iterations"] = 1000
        config["use_img_perspectives"] = False
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
    elif exp_name == "one-hot-no-ss":
        config["max_iterations"] = 1000
        config["use_img_perspectives"] = False
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
        config["no_ss"] = True
    elif exp_name == "base-no-ss":
        config["max_iterations"] = 10000
        config["use_img_perspectives"] = True
        config["nb_agents"] = 2
        config["nb_features"] = 5
        config["ood"] = False
        config["no_ss"] = True
    else:
        assert (False)

    ''' Experiment Seed & Name '''
    config["seed"] = seed
    config["exp_name"] = exp_name
    config["max_iterations"] = opts.max_iterations

    ''' Experiment path '''
    exp_path = os.path.join(path, 'results', config["exp_name"] + "/")
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    ''' Launch Experiment '''
    launch_exp(config)

    ''' Evaluate Population '''
    if config["no_ss"]:
        eval_ablation(exp_name, seed)
    else:
        eval(exp_name, seed)
