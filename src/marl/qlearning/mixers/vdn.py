from rlenv import RLEnv
import torch

from marl.models.nn import Mixer


class VDN(Mixer):
    def forward(self, qvalues: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return qvalues.sum(dim=-1)

    def save(self, directory: str):
        return

    def load(self, directory: str):
        return

    @classmethod
    def from_env(cls, env: RLEnv):
        return VDN(env.n_agents)
