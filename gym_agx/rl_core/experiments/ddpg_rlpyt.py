import os
import argparse
import torch

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from gym_agx.rl_core.gym import make as gym_make
from gym_agx.rl_core.algos.ddpg import DDPG
from gym_agx.rl_core.agents.ddpg_agent import DdpgAgent
from gym_agx.rl_core.models.mlp import MuMlpModel


def build_and_train(env_id="BendWire-v0", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuSampler
        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")
    else:
        print("No GPUs available!")

    sampler = Sampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        # Sampler batch size: batch_T * batch_B
        batch_T=2,  # Time-steps per sampler iteration.
        batch_B=n_parallel,  # Parallel environments (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=2,
        eval_max_steps=int(1e4),
        eval_max_trajectories=5,
    )
    algo = DDPG()  # Run with defaults.
    agent = DdpgAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=2e4,
        log_interval_steps=2e3,
        affinity=affinity,
    )

    config = dict(env_id=env_id)
    name = sample_mode + "_ddpg_" + env_id
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(dir_path, f'runs/rlpyt/{env_id}/')
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


def test(env_id, run_ID):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(dir_path, f'runs/rlpyt/{env_id}/run_{run_ID}/params.pkl')
    state_dict = torch.load(model_dir)
    agent_state_dict = state_dict['agent_state_dict']
    env = gym_make(env_id)
    obs = env.reset()
    agent = MuMlpModel(observation_shape=obs.shape, action_size=env.action_space.shape[0], hidden_sizes=[8, 16])
    agent.load_state_dict(agent_state_dict["model"])

    for _ in range(3):
        obs = env.reset()
        action = env.action_space.sample() * 0
        reward = 0
        for _ in range(2000):
            env.render()
            action = agent(torch.from_numpy(obs), action, reward)
            obs, reward, done, info = env.step(action.detach().numpy())
            if done:
                if info.is_success:
                    print("Goal reached!", "reward=", reward)
                else:
                    print("Stopped early.")
                break
    env.close()


def main(train, test_ID=None):
    env_id = 'BendWireDense-v0'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(dir_path, f'runs/rlpyt/{env_id}/')
    runs = os.listdir(log_dir)
    runs.sort()
    for i in range(len(runs)):
        runs[i] = int(runs[i].split('_')[1])

    run_ID = runs[-1]

    if train:
        run_ID += 1

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--env_id', help='environment ID', default=env_id)
        parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=run_ID)
        parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
        parser.add_argument('--sample_mode', help='serial or parallel sampling',
                            type=str, default='cpu', choices=['serial', 'cpu', 'gpu', 'alternating'])
        parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=4)
        args = parser.parse_args()
        build_and_train(
            env_id=args.env_id,
            run_ID=args.run_ID,
            cuda_idx=args.cuda_idx,
            sample_mode=args.sample_mode,
            n_parallel=args.n_parallel,
        )

    if not train and test_ID is not None:
        run_ID = test_ID
    test(env_id, run_ID)


if __name__ == "__main__":
    main(False, 0)
