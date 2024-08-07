from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Any
from typing_extensions import Self
from rlenv import Transition, Episode

import torch


@dataclass
class Trainer(ABC):
    """Algorithm trainer class. Needed to train an algorithm but not to test it."""

    name: str
    update_interval: int
    """
    How often to update the algorithm. 
    If the algorithm is trained on episodes, this is the number of episodes between each update.
    If the algorithm is trained on steps, this is the number of steps between each update.
    """
    update_on_steps: bool
    """Whether to update on steps."""
    update_on_episodes: bool
    """Whether to update on episodes."""

    def __init__(self, update_type: Literal["step", "episode"], update_interval: int):
        assert update_type in ["step", "episode"]
        assert update_interval > 0
        self.name = self.__class__.__name__
        self.update_interval = update_interval
        self.update_on_steps = update_type == "step"
        self.update_on_episodes = update_type == "episode"

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        """
        Update to call after each step. Should be run when update_after_each == "step".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Update to call after each episode. Should be run when update_after_each == "episode".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    @abstractmethod
    def to(self, device: torch.device) -> Self:
        """Send the tensors to the given device."""

    @abstractmethod
    def randomize(self):
        """Randomize the state of the trainer."""
