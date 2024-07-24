import os
import json
from datetime import datetime
from rlenv import Episode
from typing import Optional
from dataclasses import dataclass
from marl.models.algo import RLAlgo
from marl import logging


TRAIN = "train.csv"
TEST = "test.csv"
TRAINING_DATA = "training_data.csv"
ENV_PICKLE = "env.pkl"
ACTIONS = "actions.json"
PID = "pid"

# Dataframe columns
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"


@dataclass
class Run:
    rundir: str
    seed: int

    def __init__(self, rundir: str):
        """This constructor is not meant to be called directly. Use static methods `create` and `load` instead."""
        self.rundir = rundir
        self.seed = int(os.path.basename(rundir).split("=")[1])

    @staticmethod
    def create(logdir: str, seed: int):
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        rundir = os.path.join(logdir, f"run_{now}_seed={seed}")
        os.makedirs(rundir, exist_ok=False)
        return Run(rundir)

    def test_dir(self, time_step: int, test_num: Optional[int] = None):
        test_dir = os.path.join(self.rundir, "test", f"{time_step}")
        if test_num is not None:
            test_dir = os.path.join(test_dir, f"{test_num}")
        return test_dir

    @property
    def test_filename(self):
        return os.path.join(self.rundir, TEST)

    @property
    def train_filename(self):
        return os.path.join(self.rundir, TRAIN)

    @property
    def training_data_filename(self):
        return os.path.join(self.rundir, TRAINING_DATA)

    def __enter__(self):
        return RunHandle(
            train_logger=logging.CSVLogger(self.train_filename),
            test_logger=logging.CSVLogger(self.test_filename),
            training_data_logger=logging.CSVLogger(self.training_data_filename),
            run=self,
        )

    def __exit__(self, *args):
        pass


class RunHandle:
    def __init__(
        self,
        train_logger: logging.CSVLogger,
        test_logger: logging.CSVLogger,
        training_data_logger: logging.CSVLogger,
        run: Run,
    ):
        self.train_logger = train_logger
        self.test_logger = test_logger
        self.training_data_logger = training_data_logger
        self.run = run

    def log_tests(self, episodes: list[Episode], algo: RLAlgo, time_step: int):
        algo.save(self.run.test_dir(time_step))
        for i, episode in enumerate(episodes):
            episode_directory = self.run.test_dir(time_step, i)
            self.test_logger.log(episode.metrics, time_step)
            os.makedirs(episode_directory)
            with open(os.path.join(episode_directory, ACTIONS), "w") as a:
                json.dump(episode.actions.tolist(), a)

    def log_train_episode(self, episode: Episode, time_step: int, training_logs: dict[str, float]):
        self.train_logger.log(episode.metrics, time_step)
        self.training_data_logger.log(training_logs, time_step)

    def log_train_step(self, training_logs: dict[str, float], time_step: int):
        self.training_data_logger.log(training_logs, time_step)
