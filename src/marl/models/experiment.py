import os
import pickle
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from rlenv.models import RLEnv

from marl.utils import exceptions

from .algo import RLAlgo
from .runners import SimpleRunner
from .trainer import Trainer


@dataclass
class Experiment:
    logdir: str
    algo: RLAlgo
    trainer: Trainer
    env: RLEnv
    test_interval: int
    n_steps: int
    creation_timestamp: int
    test_env: RLEnv

    def __init__(
        self,
        logdir: str,
        algo: RLAlgo,
        trainer: Trainer,
        env: RLEnv,
        test_interval: int,
        n_steps: int,
        creation_timestamp: int,
        test_env: RLEnv,
    ):
        self.logdir = logdir
        self.trainer = trainer
        self.algo = algo
        self.env = env
        self.test_interval = test_interval
        self.n_steps = n_steps
        self.creation_timestamp = creation_timestamp
        self.test_env = test_env

    @staticmethod
    def create(
        logdir: str,
        algo: RLAlgo,
        trainer: Trainer,
        env: RLEnv,
        n_steps: int,
        test_interval: int,
        test_env: Optional[RLEnv] = None,
    ) -> "Experiment":
        """Create a new experiment."""
        if test_env is not None:
            RLEnv.assert_same_inouts(env, test_env)
        else:
            test_env = deepcopy(env)

        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)

            # Remove the test and debug logs
        if logdir in ["logs/test", "logs/debug", "logs/tests"]:
            try:
                shutil.rmtree(logdir)
            except FileNotFoundError:
                pass
        try:
            os.makedirs(logdir, exist_ok=False)
            experiment = Experiment(
                logdir,
                algo=algo,
                trainer=trainer,
                env=env,
                n_steps=n_steps,
                test_interval=test_interval,
                creation_timestamp=int(time.time() * 1000),
                test_env=test_env,
            )
            experiment.save()
            return experiment
        except FileExistsError:
            raise exceptions.ExperimentAlreadyExistsException(logdir)
        except Exception as e:
            # In case the experiment could not be created for another reason, do not create the experiment and remove its directory
            shutil.rmtree(logdir, ignore_errors=True)
            raise e

    @staticmethod
    def load(logdir: str) -> "Experiment":
        """Load an experiment from disk."""
        with open(os.path.join(logdir, "experiment.pkl"), "rb") as f:
            experiment: Experiment = pickle.load(f)
        return experiment

    def save(self):
        """Save the experiment to disk."""
        os.makedirs(self.logdir, exist_ok=True)
        with open(os.path.join(self.logdir, "experiment.pkl"), "wb") as f:
            pickle.dump(self, f)

    def create_runner(self):
        return SimpleRunner(
            env=self.env,
            algo=self.algo,
            trainer=self.trainer,
            test_interval=self.test_interval,
            n_steps=self.n_steps,
            test_env=self.test_env,
        )

    @property
    def train_dir(self):
        return os.path.join(self.logdir, "train")

    @property
    def test_dir(self):
        return os.path.join(self.logdir, "test")
