# Intrinsic motivation for robotic manipulation learning with sparse rewards

As my undergraduate thesis, I studied the impact of curiosity and intrinsic motivation as exploration strategy for deep reinforcement learning agents on sparse-reward robotic manipulator environments. We found that this approach encourages increasing exploratory behaviors even after the goal tasks were learned. Furthermore, we found that adding information about other objects' states into the agent's observation is crucial for learning complex behaviors when no dense reward signal is provided.

To read the full report, [click here](https://github.com/bryanlincoln/undergraduate-thesis/blob/master/Text%20-%20Intrinsic%20motivation%20for%20robotic%20manipulation%20learning%20with%20sparse%20rewards.pdf) (Portuguese).

## Requirements

-   [Python 3](https://docs.python.org/)
-   [PyTorch](http://pytorch.org/)
-   [OpenAI Gym](https://github.com/openai/gym)
-   [OpenAI baselines](https://github.com/openai/baselines)
-   [Gym Fetch](https://github.com/jmichaux/gym-fetch)

## Usage

To run the code, simply execute `python main.py` after installing all the requirements. There are many customizable hyperparemeters and configurations. You can see them with `python main.py --help`. The exact hyperparameters for this study's experiments can be found in the folder `experiments`.

## Credits

This code was based on and adapted from

-   [Jon Michaux's implementation of intrinsic motivation](https://github.com/jmichaux/intrinsic-motivation)
-   [Santhosh Ramakrishnan's implementation of curiosity](https://github.com/srama2512/curiosity-driven-exploration)
-   [Ilya Kostrikov' implementation of recent RL algorithms](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
-   [OpenAI Baselines](https://github.com/openai/baselines)
-   [Chip Schaff's Deep Learning Library](https://github.com/cbschaff/pytorch-dl)

The inspiration and theoretic background was mainly based on

- [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/)
- [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/)

## Results

### Learned policies

Pick And Place Task (left), Push Task (center) and Reach (right).

<img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/pick.gif" width="280" height="180"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/push.gif" width="280" height="180"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/reach.gif" width="280" height="180">

### Success Rate Charts

Pick And Place Task (left), Push Task (center) and Reach (right). Blue lines are results for vanilla PPO (baseline) and red lines for PPO + intrinsic motivation.

<img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/pick.png" width="280" height="200"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/push.png" width="280" height="200"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/reach.png" width="280" height="200">

### Entropy Charts

Pick And Place Task (left), Push Task (center) and Reach (right). Blue lines are results for vanilla PPO (baseline) and red lines for PPO + intrinsic motivation.

<img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/pick_ent.png" width="280" height="200"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/push_ent.png" width="280" height="200"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/reach_ent.png" width="280" height="200">

### Intrinsic Reward Charts

Pick And Place Task (left), Push Task (center) and Reach (right).

<img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/pick_int.png" width="280" height="200"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/push_int.png" width="280" height="200"> <img src="https://github.com/bryanlincoln/undergraduate-thesis/blob/master/fig/preview/reach_int.png" width="280" height="200">
