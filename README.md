# HyperX

Code for the paper "[Exploration in Approximate Hyper-State Space for Meta Reinforcement Learning](https://arxiv.org/abs/2010.01062)" -
Luisa Zintgraf, Leo Feng, Cong Lu, Maximilian Igl, 
Kristian Hartikainen, Katja Hofmann, Shimon Whiteson, 
published at ICML 2021.

```
@inproceedings{zintgraf2021hyperx,
  title={Exploration in Approximate Hyper-State Space for Meta Reinforcement Learning},
  author={Zintgraf, Luisa and Feng, Leo and Lu, Cong and Igl, Maximilian and Hartikainen, Kristian and Hofmann, Katja and Whiteson, Shimon},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}}
```

> ! Important !
>
> If you use this code with your own environments,
> make sure to not use `np.random` in them
> (e.g. to generate the tasks) because it is not thread safe
> (and not using it may cause duplicates across threads).
> Instead, use the python native random function.
> For an example see
> [here](https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/ant_goal.py#L38).

### Requirements

We use PyTorch for this code, and log results using TensorboardX.

The main requirements can be found in `requirements.txt`.

For the MuJoCo experiments, you need to install MuJoCo.
Make sure you have the right MuJoCo version: 
For the Cheetah and Ant environments, use `mujoco150`.
(You can also use `mujoco200` except for AntGoal,
because there's a bug which leads to 80% of the env state being zero).

### Code Structure 

The main training loop is in `metalearner.py`. 
The models are in `/models/`, 
the code for the exploration bonuses in `/exploration/`,
the RL algorithms in `/algorithms/`,
and the VAE in `vae.py`.

### Running experiments

To run the experiments found in the paper, execute these commands:
- Mountain Treasure:\
    `python main.py --env-type treasure_hunt_hyperx`
- Multi-Stage GridWorld:\
  `python main.py --env-type room_hyperx`
- Sparse HalfCheetahDir:\
  `python main.py --env-type cds_hyperx`
- Sparse AntGoal:\
  `python main.py --env-type sparse_ant_goal_hyperx`
- 2D Navigation Point Robot: \
  `python main.py --env-type pointrobot_sparse_hyperx`

Additional experiments, in particular baselines, are listed in `main.py`.

The results will by default be saved at `./logs`, 
but you can also pass a flag with an alternative directory using `--results_log_dir /path/to/dir`.
Results will be written to tensorboard event files,
and some visualisations will be printed now and then.
