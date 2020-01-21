import argparse
from collections import deque
import os
import random
import sys
import time
import yaml
import types
import numpy as np
import gym
import torch
import torch.optim as optim

from baselines.common.vec_env.vec_normalize import VecNormalize
from models import ActorCritic, FwdDyn
from ppo import PPO
import gym_fetch
from wrappers import make_vec_envs
import utils

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--num-evals', type=int, default=100)
parser.add_argument('--num-processes', type=int, default=1)
parser.add_argument('--checkpoint', type=str, default='run13')
parser.add_argument('--experiment-name', type=str, default='RandomAgent')
parser.add_argument('--env-id', type=str, default='FetchPush-v1')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--clean-dir', action='store_true')
parser.add_argument('--add-intrinsic-reward', action='store_true')
parser.add_argument('--num-steps', type=int, default=2048)
parser.add_argument('--ppo-epochs', type=int, default=10)
parser.add_argument('--num-mini-batch', type=int, default=32)
parser.add_argument('--pi-lr', type=float, default=1e-4)
parser.add_argument('--v-lr', type=float, default=1e-3)
parser.add_argument('--dyn-lr', type=float, default=1e-3)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--clip-param', type=float, default=0.3)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--grad-norm-max', type=float, default=5.0)
parser.add_argument('--dyn-grad-norm-max', type=float, default=5)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--use-gae', action='store_true')
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--share-optim', action='store_true')
parser.add_argument('--predict-delta-obs', action='store_true')
parser.add_argument('--use-linear-lr-decay', action='store_true')
parser.add_argument('--use-clipped-value-loss', action='store_true')
parser.add_argument('--use-tensorboard', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-render', action='store_true', default=False)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    args.cuda = False
    args.render = not args.no_render

    # set device and random seeds
    device = torch.device("cpu")
    torch.set_num_threads(1)
    utils.set_random_seeds(args.seed, args.cuda, args.debug)

    # setup environment
    envs = make_vec_envs(env_id=args.env_id,
                         seed=args.seed,
                         num_processes=args.num_processes,
                         gamma=None,
                         log_dir=None,
                         device=device,
                         obs_keys=['observation', 'desired_goal'] if not args.env_id.startswith(
                             'metaworld') else None,
                         allow_early_resets=True,
                         max_steps=args.num_steps,
                         evaluating=True)

    # create agent
    checkpoint_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        args.log_dir,
        args.env_id,
        args.experiment_name,
        args.checkpoint
    )

    agent = PPO(None,
            envs.observation_space,
            envs.action_space,
            actor_critic=ActorCritic,
            dynamics_model=FwdDyn,
            optimizer=optim.Adam,
            hidden_size=args.hidden_size,
            num_steps=args.num_steps,
            num_processes=args.num_processes,
            ppo_epochs=args.ppo_epochs,
            num_mini_batch=args.num_mini_batch,
            pi_lr=args.pi_lr,
            v_lr=args.v_lr,
            dyn_lr=args.dyn_lr,
            clip_param=args.clip_param,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            grad_norm_max=args.grad_norm_max,
            use_clipped_value_loss=True,
            use_tensorboard=args.use_tensorboard,
            add_intrinsic_reward=args.add_intrinsic_reward,
            predict_delta_obs=args.predict_delta_obs,
            device=device,
            share_optim=args.share_optim,
            debug=None)

    ob_rms = agent.load_checkpoint(checkpoint_dir)
    agent.eval()

    # set same statistics for normalization as in training
    if ob_rms is not None and isinstance(envs.venv, VecNormalize):
        envs.venv.ob_rms = ob_rms

    # start testing
    start = time.time()

    for trial in range(args.num_evals):
        print('Trial ', trial, 'of', args.num_evals)
        
        obs = envs.reset()
        agent.rollouts.obs[0].copy_(obs[1])
        agent.rollouts.to(device)

        extrinsic_rewards = []
        episode_length = []
        solved_episodes = []

        for step in range(args.num_steps):
            #time.sleep(0.02)

            # render
            if args.render:
                envs.render()

            # select action
            value, action, action_log_probs = agent.select_action(step)

            # take a step in the environment
            obs, reward, done, infos = envs.step(action)

            # store experience
            agent.store_rollout(obs[1], action, action_log_probs,
                                value, reward, torch.tensor(0).view(1, 1), done)

            # get final episode rewards
            for info in infos:
                if 'episode' in info.keys():
                    extrinsic_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])
                    solved_episodes.append(info['is_success'])

        print('Extrinsic/Mean', np.mean(extrinsic_rewards))
        print('Extrinsic/Median', np.median(extrinsic_rewards))
        print('Extrinsic/Min', np.min(extrinsic_rewards))
        print('Extrinsic/Max', np.max(extrinsic_rewards))
        print('Episodes/Solved', np.mean(solved_episodes))
        print('Episodes/Length', np.mean(episode_length))
        print()
