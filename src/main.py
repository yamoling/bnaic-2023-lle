import torch
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import marl
import rlenv
from marl.utils import Schedule
from lle import LLE, ObservationType
from marl.training import DQNTrainer
from marl.training.qtarget_updater import SoftUpdate


@dataclass
class Parameters:
    memory_size: int = 50_000
    rnd: bool = False
    gamma: float = 0.95
    lr: float = 5e-4
    batch_size: int = 64
    per: bool = False
    grad_norm_clipping: float = 10.0
    train_interval: tuple[int, Literal["step", "episode"]] = (5, "step")
    mixer: Literal["vdn", "qmix"] | None = "vdn"
    optimizer: Literal["adam", "rmsprop"] = "adam"
    n_steps: int = 1_000_000
    test_interval: int = 5000


def create_experiment(params: Parameters):
    # Create LLE
    env = LLE.level(6).obs_type(ObservationType.LAYERED).state_type(ObservationType.STATE).build()
    # Wrap the environment to limit the time and add the agent id to the observations
    env = rlenv.Builder(env).agent_id().time_limit(78).build()

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    if params.per:
        memory = marl.models.PrioritizedMemory(
            memory=memory,
            alpha=0.6,
            beta=Schedule.linear(0.5, 1.0, int(1e6)),
        )
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=500_000,
    )
    match params.mixer:
        case "vdn":
            mixer = marl.qlearning.VDN.from_env(env)
        case "qmix":
            mixer = marl.qlearning.QMix.from_env(env)
        case None:
            mixer = None
        case other:
            raise ValueError(f"Unknown mixer: {other}")

    if params.rnd:
        ir_module = marl.intrinsic_reward.RandomNetworkDistillation(
            target=marl.nn.model_bank.CNN(
                input_shape=env.observation_shape,
                extras_size=env.extra_feature_shape[0],
                output_shape=(env.reward_size, 512),
            ),
            normalise_rewards=False,
        )
    else:
        ir_module = None

    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        optimiser=params.optimizer,
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=params.lr,
        batch_size=params.batch_size,
        train_interval=params.train_interval,
        gamma=params.gamma,
        mixer=mixer,
        grad_norm_clipping=params.grad_norm_clipping,
        ir_module=ir_module,
    )

    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    logdir = datetime.now().strftime("logs/%Y-%m-%d-%H-%M-%S")
    exp = marl.Experiment.create(
        logdir,
        algo,
        trainer,
        env,
        params.n_steps,
        params.test_interval,
    )

    run_experiment(exp)


def run_experiment(exp: marl.Experiment):
    N_RUNS = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting {N_RUNS} runs")
    for run_num in range(N_RUNS):
        runner = exp.create_runner().to(device)
        runner.run(exp.logdir, run_num, 1, quiet=False)


if __name__ == "__main__":
    # Default parameters if no argument is provided
    params = Parameters()
    create_experiment(params)
