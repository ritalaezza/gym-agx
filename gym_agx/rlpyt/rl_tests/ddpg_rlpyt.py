"""
Runs one instance of the BendWire environment and optimizes using DDPG algorithm.
Can choose between configurations for use of CPU/GPU for sampling (serial or parallel) and optimization (serial).

The kwarg snapshot_mode="last" to logger context will save the latest model at
every log point (see inside the logger for other options).

In viskit, whatever (nested) key-value pairs appear in config will become plottable
keys for showing several experiments.  If you need to add more after an experiment,
use rlpyt.utils.logging.context.add_exp_param().

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from gym_agx.rlpyt.gym import make as gym_make
from gym_agx.rlpyt.algos.ddpg import DDPG
from gym_agx.rlpyt.agents.ddpg_agent import DdpgAgent


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
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=100,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DDPG()  # Run with defaults.
    agent = DdpgAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e4,
        affinity=affinity,
    )
    config = dict(env_id=env_id)
    name = sample_mode + "_ddpg_" + env_id
    log_dir = "rl_tests"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='BendWireDense-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
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
