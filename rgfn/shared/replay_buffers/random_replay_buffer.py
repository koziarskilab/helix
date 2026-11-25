from typing import Hashable, Iterator, List, Set, TypeVar

import gin
import numpy as np
from more_itertools import chunked

from rgfn.api.replay_buffer_base import ReplayBufferBase
from rgfn.api.sampler_base import SamplerBase
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace

THashableState = TypeVar("THashableState", bound=Hashable)


@gin.configurable()
class RandomReplayBuffer(ReplayBufferBase[THashableState, TActionSpace, TAction]):
    def __init__(
        self,
        sampler: SamplerBase[THashableState, TActionSpace, TAction],
        max_size: int = int(1e6),
    ):
        super().__init__(sampler)
        self.max_size = int(max_size)
        self.states_list: List[THashableState] = []
        self.states_set: Set[THashableState] = set()

    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories[THashableState, TActionSpace, TAction]]:
        n_total_trajectories = min(n_total_trajectories, self.size)
        batch_size = n_total_trajectories if batch_size == -1 else batch_size

        if n_total_trajectories == 0:
            yield Trajectories()
            return

        if n_total_trajectories >= self.size:
            sampled_states = self.states_list
        else:
            sampled_states = np.random.choice(
                self.states_list, size=n_total_trajectories, replace=False
            )

        for sampled_states_chunk in chunked(sampled_states, batch_size):
            yield self.sampler.sample_trajectories_from_sources(sampled_states_chunk)

    def add_trajectories(self, trajectories: Trajectories[THashableState, TActionSpace, TAction]):
        terminal_states = trajectories.get_last_states_flat()

        for state in terminal_states:
            if state not in self.states_set:
                self.states_list.append(state)
                self.states_set.add(state)
                if self.size > self.max_size:
                    el = self.states_list.pop(0)
                    self.states_set.remove(el)

    def state_dict(self) -> dict:
        return {"states_list": self.states_list}

    def load_state_dict(self, state_dict: dict):
        self.states_list = state_dict["states_list"]
        self.states_set = set(self.states_list)

    @property
    def size(self):
        return len(self.states_list)
