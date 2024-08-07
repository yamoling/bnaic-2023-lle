from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional
from rlenv import RLEnv
import torch
from marl.models.algo import RLAlgo
from marl.models.trainer import Trainer


class Runner(ABC):
    def __init__(self, algo: RLAlgo, env: RLEnv, trainer: Trainer, test_env: Optional[RLEnv] = None):
        self._trainer = trainer
        self._env = env
        self._algo = algo
        if test_env is None:
            test_env = deepcopy(env)
        self._test_env = test_env

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def to(self, device: torch.device):
        self._algo.to(device)
        self._trainer.to(device)
        return self

    def randomize(self):
        self._algo.randomize()
        self._trainer.randomize()
