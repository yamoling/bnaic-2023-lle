import torch

from typing import Any, Literal, Optional
from copy import deepcopy
from rlenv import Transition, Episode
from marl.models import QNetwork, Mixer, ReplayMemory, Trainer, Policy, PrioritizedMemory
from marl.models.batch import Batch
from marl.intrinsic_reward import IRModule
from .qtarget_updater import TargetParametersUpdater, SoftUpdate
from marl.utils import defaults_to

from dataclasses import dataclass


@dataclass
class DQNTrainer(Trainer):
    qnetwork: QNetwork
    policy: Policy
    memory: ReplayMemory[Batch, Episode | Transition]
    gamma: float
    batch_size: int
    target_updater: TargetParametersUpdater
    double_qlearning: bool
    mixer: Optional[Mixer]
    ir_module: Optional[IRModule]
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        qnetwork: QNetwork,
        train_policy: Policy,
        memory: ReplayMemory,
        mixer: Optional[Mixer],
        gamma: float = 0.99,
        batch_size: int = 64,
        lr: float = 1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        target_updater: Optional[TargetParametersUpdater] = None,
        double_qlearning: bool = False,
        train_interval: tuple[int, Literal["step", "episode"]] = (5, "step"),
        ir_module: Optional[IRModule] = None,
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__(train_interval[1], train_interval[0])
        self.qnetwork = qnetwork
        self.qtarget = deepcopy(qnetwork)
        self.device = qnetwork.device
        self.policy = train_policy
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_updater = defaults_to(target_updater, lambda: SoftUpdate(1e-2))
        self.double_qlearning = double_qlearning
        self.mixer = mixer
        self.target_mixer = deepcopy(mixer)
        self.ir_module = ir_module
        self.update_num = 0

        # Parameters and optimiser
        self.grad_norm_clipping = grad_norm_clipping
        self.target_updater.add_parameters(qnetwork.parameters(), self.qtarget.parameters())
        if mixer is not None and self.target_mixer is not None:
            self.target_updater.add_parameters(mixer.parameters(), self.target_mixer.parameters())
        match optimiser:
            case "adam":
                self.optimiser = torch.optim.Adam(self.target_updater.parameters, lr=lr)
            case "rmsprop":
                self.optimiser = torch.optim.RMSprop(self.target_updater.parameters, lr=lr, eps=1e-5)
            case other:
                raise ValueError(f"Unknown optimiser: {other}. Expected 'adam' or 'rmsprop'.")

    def _update(self, time_step: int):
        self.update_num += 1
        if self.update_num % self.update_interval != 0 or not self._can_update():
            return {}
        logs, td_error = self.optimise_qnetwork()
        logs = logs | self.policy.update(time_step)
        logs = logs | self.target_updater.update(time_step)
        if isinstance(self.memory, PrioritizedMemory):
            logs = logs | self.memory.update(td_error)
        if self.ir_module is not None:
            logs = logs | self.ir_module.update(time_step)
        return logs

    def _can_update(self):
        return self.memory.can_sample(self.batch_size)

    def _next_state_value(self, batch: Batch):
        # We use the all_obs_ and all_extras_ to handle the case of recurrent qnetworks that require the first element of the sequence.
        next_qvalues = self.qtarget.batch_forward(batch.all_obs_, batch.all_extras_)[1:].squeeze(-1)
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            qvalues_for_index = self.qnetwork.batch_forward(batch.all_obs_, batch.all_extras_)[1:].squeeze(-1)
        else:
            qvalues_for_index = next_qvalues
        qvalues_for_index[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        # indices = indices.unsqueeze(-1).repeat(*(1 for _ in indices.shape), batch.reward_size)
        next_values = torch.gather(next_qvalues, -1, indices).squeeze(-1)
        if self.target_mixer is not None:
            mixed_next_values = self.target_mixer.forward(next_values, batch.states_, batch.one_hot_actions, next_qvalues)
        else:
            mixed_next_values = next_values
        return mixed_next_values

    def optimise_qnetwork(self):
        # Note: there are many x.squeeze(-1) because the code was initially thought with multi-objective RL in mind.
        batch = self.memory.sample(self.batch_size).to(self.qnetwork.device)
        batch.rewards = batch.rewards.squeeze(-1)
        batch.masks = batch.masks.squeeze(-1)
        batch.dones = batch.dones.squeeze(-1)
        # batch.multi_objective()
        if self.ir_module is not None:
            assert self.mixer is not None, "Intrinsic rewards are not supported for independent learners."
            batch.rewards = batch.rewards + self.ir_module.compute(batch).squeeze(-1)

        # Qvalues and qvalues with target network computation
        qvalues = self.qnetwork.batch_forward(batch.obs, batch.extras).squeeze(-1)
        chosen_qvalues = torch.gather(qvalues, dim=-1, index=batch.actions).squeeze(-1)
        if self.mixer is not None:
            mixed_qvalues = self.mixer.forward(chosen_qvalues, batch.states, batch.one_hot_actions, qvalues).squeeze(-1)
        else:
            batch = batch.for_individual_learners()
            mixed_qvalues = chosen_qvalues

        # Drop variables to prevent using them mistakenly
        del qvalues
        del chosen_qvalues

        # Qtargets computation
        next_values = self._next_state_value(batch).squeeze(-1)
        assert batch.rewards.shape == next_values.shape == batch.dones.shape == mixed_qvalues.shape == batch.masks.shape
        qtargets = batch.rewards + self.gamma * next_values * (1 - batch.dones)
        # Compute the loss
        td_error = mixed_qvalues - qtargets.detach()
        td_error = td_error * batch.masks
        squared_error = td_error**2
        if batch.importance_sampling_weights is not None:
            assert self.mixer is not None, "PER is not supported for independent learners."
            if squared_error.shape != batch.importance_sampling_weights.shape:
                batch.importance_sampling_weights = batch.importance_sampling_weights.reshape_as(squared_error)
            squared_error = squared_error * batch.importance_sampling_weights
        loss = squared_error.sum() / batch.masks.sum()

        # Optimize
        logs = {"loss": float(loss.item())}
        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), self.grad_norm_clipping)
            logs["grad_norm"] = grad_norm.item()
        self.optimiser.step()
        return logs, td_error

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        if self.memory.update_on_transitions:
            self.memory.add(transition)
        if not self.update_on_steps:
            return {}
        return self._update(time_step)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        if self.memory.update_on_episodes:
            self.memory.add(episode)
        if not self.update_on_episodes:
            return {}
        return self._update(time_step)

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.qtarget.to(device)
        if self.mixer is not None:
            self.mixer.to(device)
        if self.target_mixer is not None:
            self.target_mixer.to(device)
        if self.ir_module is not None:
            self.ir_module.to(device)
        self.device = device
        return self

    def randomize(self):
        self.qnetwork.randomize()
        self.qtarget.randomize()
        if self.mixer is not None:
            self.mixer.randomize()
        if self.target_mixer is not None:
            self.target_mixer.randomize()
        if self.ir_module is not None:
            self.ir_module.randomize()
