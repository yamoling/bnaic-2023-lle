from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterable, Literal, TypeVar

import numpy as np
from rlenv import Episode, Transition

from marl.models.batch import Batch, TransitionBatch


T = TypeVar("T")
B = TypeVar("B", bound=Batch)


@dataclass
class ReplayMemory(Generic[B, T], ABC):
    """Parent class of any ReplayMemory"""

    max_size: int
    name: str
    update_on_transitions: bool
    update_on_episodes: bool

    def __init__(self, max_size: int, update_on: Literal["transition", "episode"]):
        self._memory: Deque[T] = deque(maxlen=max_size)
        self.max_size = max_size
        self.name = self.__class__.__name__
        self.update_on_transitions = update_on == "transition"
        self.update_on_episodes = update_on == "episode"

    def add(self, item: T):
        """Add an item (transition, episode, ...) to the memory"""
        self._memory.append(item)

    def sample(self, batch_size: int) -> B:
        """Sample the memory to retrieve a `Batch`"""
        indices = np.random.randint(0, len(self), batch_size)
        return self.get_batch(indices)

    def can_sample(self, batch_size: int) -> bool:
        """Return whether the memory contains enough items to sample a batch of the given size"""
        return len(self) >= batch_size

    def clear(self):
        self._memory.clear()

    @property
    def is_full(self):
        return len(self) == self.max_size

    @abstractmethod
    def get_batch(self, indices: Iterable[int]) -> B:
        """Create a `Batch` from the given indices"""

    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self, index: int) -> T:
        return self._memory[index]


class TransitionMemory(ReplayMemory[TransitionBatch, Transition]):
    """Replay Memory that stores Transitions"""

    def __init__(self, max_size: int):
        super().__init__(max_size, "transition")

    def get_batch(self, indices: Iterable[int]):
        transitions = [self._memory[i] for i in indices]
        return TransitionBatch(transitions)
