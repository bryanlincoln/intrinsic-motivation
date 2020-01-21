import os
import sys
import shutil
import random
import numpy as np
import torch
import torch.nn as nn


def create_log_dirs(exp_name, checkpoint=False, tbx=False, force_clean=False, logsdir='logs'):
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), logsdir)
    exp_dir = os.path.join(basedir, exp_name)
    try:
        os.makedirs(exp_dir)
    except OSError:
        if force_clean:
            # remove all files
            for file in os.listdir(exp_dir):
                file_path = os.path.join(exp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except:
                    pass
    run_dir = 'run' + str(sum(os.path.isdir(os.path.abspath(os.path.join(exp_dir, i)))
                              for i in os.listdir(exp_dir)))
    run_dir = os.path.join(exp_dir, run_dir)
    # make base dir
    os.makedirs(run_dir)
    # make monitor dir
    os.makedirs(os.path.join(run_dir, 'monitor'))
    # make checkpoint dir
    if checkpoint:
        os.makedirs(os.path.join(run_dir, 'checkpoint'))
    # make tensorboard dir
    if tbx:
        os.makedirs(os.path.join(run_dir, 'tensorboard'))
    # make eval dir
    eval_dir = os.path.join(run_dir, 'eval')
    os.makedirs(eval_dir)
    os.makedirs(os.path.join(eval_dir, 'video'))
    os.makedirs(os.path.join(eval_dir, 'monitor'))
    return run_dir


def set_random_seeds(seed, cuda=True, debug=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda and torch.cuda.is_available() and debug:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_device(cuda, num_threads=1):
    device = torch.device(
        "cuda:0" if cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(num_threads)
    return device


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def compute_intrinsic_rewards(model, obs, action, next_obs, eta=0.75):
    next_obs_preds = model(obs, action)
    # return 0.5 * eta * torch.norm(next_obs - next_obs_preds, p=2, dim=-1).pow(2).unsqueeze(-1)
    return 0.5 * eta * (next_obs_preds - next_obs).pow(2).sum(-1).unsqueeze(-1)


def init_tanh(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                              constant_(x, 0), 5.0/3)


def init_relu(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                              constant_(x, 0), np.sqrt(2))


def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                          constant_(x, 0))


def update_linear_schedule(optimizer, update, total_num_updates, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (update / float(total_num_updates)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
