from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional

from rgfn.shared.policies.uniform_policy import IndexedActionSpaceBase

from .data_structures import AnchoredReaction, Molecule


@dataclass(frozen=True)
class ReactionStateH:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"SH"


@dataclass(frozen=True)
class ReactionActionH:
    fragment: Molecule
    budget: int
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AH({self.fragment}, {self.budget})"


@dataclass(frozen=True)
class ReactionActionSpaceH(IndexedActionSpaceBase[ReactionActionH]):
    all_actions: Tuple[ReactionActionH, ...]
    possible_actions_mask: List[bool]

    def get_action_at_idx(self, idx: int) -> ReactionActionH:
        return self.all_actions[idx]

    def get_idx_of_action(self, action: ReactionActionH) -> int:
        return action.idx

    def get_possible_actions_indices(self) -> List[int]:
        return [idx for idx, mask in enumerate(self.possible_actions_mask) if mask]

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_action_indices = self.get_possible_actions_indices()
        possible_actions = [self.all_actions[idx] for idx in possible_action_indices]
        return f"ASH({possible_actions})"


@dataclass(frozen=True)
class ReactionActionExit:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "AExit"


@dataclass(frozen=True)
class ReactionStateX:
    molecule: Molecule
    num_decompositions: int
    budget: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"SX({self.molecule}, {self.num_decompositions}, {self.budget})"


@dataclass(frozen=True)
class ReactionActionX:
    input_molecule: Molecule
    input_reaction: AnchoredReaction
    input_fragments: Tuple[Molecule, ...]
    output_molecule: Molecule

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AX({self.input_molecule}, {self.input_reaction}, {self.input_fragments}, {self.output_molecule})"


@dataclass(frozen=True)
class ReactionActionSpaceX(IndexedActionSpaceBase[ReactionActionX | ReactionActionExit]):
    possible_actions: Tuple[ReactionActionX, ...]
    exit_action: Optional[ReactionActionExit]

    def get_action_at_idx(self, idx: int) -> ReactionActionX | ReactionActionExit:
        if idx == len(self.possible_actions):
            assert self.exit_action is not None
            return self.exit_action
        else:
            return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionActionX | ReactionActionExit) -> int:
        if isinstance(action, ReactionActionExit):
            assert self.exit_action is not None
            return len(self.possible_actions)
        else:
            return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        extra = 1 if self.exit_action is not None else 0
        return list(range(len(self.possible_actions) + extra))

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_actions = self.possible_actions
        if self.exit_action is not None:
            possible_actions += (self.exit_action,)
        return f"ASX({possible_actions})"


@dataclass(frozen=True)
class ReactionStateY:
    molecule: Molecule
    anchored_reaction: AnchoredReaction
    fragments: Tuple[Molecule, ...]
    num_decompositions: int
    budget: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"SY({self.molecule}, {self.anchored_reaction}, {self.fragments}, {self.num_decompositions}, {self.budget})"
        )

@dataclass(frozen=True)
class ReactionActionY:
    fragment: Molecule
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AY({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpaceY(IndexedActionSpaceBase[ReactionActionY | ReactionActionExit]):
    possible_actions: Tuple[ReactionActionY, ...]
    exit_action: Optional[ReactionActionExit]

    def get_action_at_idx(self, idx: int) -> ReactionActionY | ReactionActionExit:
        if idx == len(self.possible_actions):
            assert self.exit_action is not None
            return self.exit_action
        else:
            return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionActionY | ReactionActionExit) -> int:
        if isinstance(action, ReactionActionExit):
            assert self.exit_action is not None
            return len(self.possible_actions)
        else:
            return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        extra = 1 if self.exit_action is not None else 0
        return list(range(len(self.possible_actions) + extra))

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_actions = self.possible_actions
        if self.exit_action is not None:
            possible_actions += (self.exit_action,)
        return f"ASY({possible_actions})"


@dataclass(frozen=True)
class ReactionStateZ:
    molecule: Molecule
    anchored_reaction: AnchoredReaction
    fragments: Tuple[Molecule, ...]
    num_decompositions: int
    budget: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"SZ({self.molecule}, {self.anchored_reaction}, {self.fragments}, {self.num_decompositions}, {self.budget})"
        )


@dataclass(frozen=True)
class ReactionActionZ:
    anchored_reaction: AnchoredReaction
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AZ({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpaceZ(IndexedActionSpaceBase[ReactionActionZ | ReactionActionExit]):
    possible_action: ReactionActionZ
    exit_action: Optional[ReactionActionExit]

    def get_action_at_idx(self, idx: int) -> ReactionActionZ | ReactionActionExit:
        if idx == 0:
            return self.possible_action
        elif idx == 1:
            assert self.exit_action is not None
            return self.exit_action
        else:
            raise ValueError("Unexpected action index.")

    def get_idx_of_action(self, action: ReactionActionZ | ReactionActionExit) -> int:
        if isinstance(action, ReactionActionZ):
            return 0
        elif isinstance(action, ReactionActionExit):
            assert self.exit_action is not None
            return 1
        else:
            raise ValueError("Unexpected action.")

    def get_possible_actions_indices(self) -> List[int]:
        return [0, 1] if self.exit_action is not None else [0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.exit_action is not None:
            return f"ASZ({self.possible_action, self.exit_action})"
        else:
            return f"SZZ({self.possible_action})"


@dataclass(frozen=True)
class ReactionActionSpaceHorZBackward(IndexedActionSpaceBase[ReactionActionH | ReactionActionZ]):
    possible_actions: Tuple[ReactionActionH | ReactionActionZ, ...]

    def get_action_at_idx(self, idx: int) -> ReactionActionH | ReactionActionZ:
        return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionActionH | ReactionActionZ) -> int:
        return self.possible_actions.index(action)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ASZBackward({self.possible_actions})"


@dataclass(frozen=True)
class ReactionState0:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "S0"


@dataclass(frozen=True)
class ReactionAction0:
    fragment: Molecule
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"A0({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpace0(IndexedActionSpaceBase[ReactionAction0]):
    all_actions: Tuple[ReactionAction0, ...]
    possible_actions_mask: List[bool]

    def get_action_at_idx(self, idx: int) -> ReactionAction0:
        return self.all_actions[idx]

    def get_idx_of_action(self, action: ReactionAction0) -> int:
        return action.idx

    def get_possible_actions_indices(self) -> List[int]:
        return [idx for idx, mask in enumerate(self.possible_actions_mask) if mask]

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_action_indices = self.get_possible_actions_indices()
        possible_actions = [self.all_actions[idx] for idx in possible_action_indices]
        return f"AS0({possible_actions})"


@dataclass(frozen=True)
class ReactionState0Invalid:
    """
    In our code there are some really rare cases of reactions (reactants, product, template) that cannot be reversed:
    they can be applied in one direction, but not in the other. The `ReactionEarlyTerminalState` is a hacky way of
    dealing with such cases in the forward sampling (a proper way is expensive). If we encounter a reaction that
    cannot be reversed, we terminate the generation process. The `ReactionState0Invalid` is a similar concept but
    for the backward sampling. In `ReactionEnv._is_decomposable` method we don't check if there are some
    irreversible reactions in the synthesis tree (it's expensive), so our backward trajectory could end up in a
    molecule that can only be decomposed by an irreversible reaction, which we cannot allow for the sake of the
    compatibility with forward sampling. At this point, we terminate the generation process and transition to
    `ReactionState0Invalid` state. All such invalid trajectories are then filtered from the trajectories batch.
    """

    previous_state: Any = field(hash=False, compare=False)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "S0Invalid"


@dataclass(frozen=True)
class ReactionAction0Invalid:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "A0Invalid"


@dataclass(frozen=True)
class ReactionActionSpace0Invalid(IndexedActionSpaceBase[ReactionAction0Invalid]):
    possible_action = ReactionAction0Invalid()

    def get_action_at_idx(self, idx: int) -> ReactionAction0Invalid:
        return self.possible_action

    def get_idx_of_action(self, action: ReactionAction0Invalid) -> int:
        return 0

    def get_possible_actions_indices(self) -> List[int]:
        return [0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "AS0Invalid"


@dataclass(frozen=True)
class ReactionStateA:
    molecule: Molecule
    num_reactions: int
    num_decompositions: int
    budget: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"SA({self.molecule}, {self.num_reactions}, {self.num_decompositions}, {self.budget})"


@dataclass(frozen=True)
class ReactionActionA:
    anchored_reaction: AnchoredReaction | None
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AA({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpaceA(IndexedActionSpaceBase[ReactionActionA | ReactionActionExit]):
    all_actions: Tuple[ReactionActionA, ...]
    possible_actions_mask: List[bool]
    exit_action: Optional[ReactionActionExit]

    def get_action_at_idx(self, idx: int) -> ReactionActionA | ReactionActionExit:
        if idx == len(self.all_actions):
            assert self.exit_action is not None
            return self.exit_action
        else:
            return self.all_actions[idx]

    def get_idx_of_action(self, action: ReactionActionA | ReactionActionExit) -> int:
        if isinstance(action, ReactionActionExit):
            assert self.exit_action is not None
            return len(self.all_actions)
        else:
            return action.idx

    def get_possible_actions_indices(self) -> List[int]:
        indices = [idx for idx, mask in enumerate(self.possible_actions_mask) if mask]
        if self.exit_action is not None:
            indices.append(len(self.all_actions))
        return indices

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_action_indices = self.get_possible_actions_indices()
        possible_actions = [self.get_action_at_idx(idx) for idx in possible_action_indices]
        return f"ASA({possible_actions})"


@dataclass(frozen=True)
class ReactionStateB:
    molecule: Molecule
    anchored_reaction: AnchoredReaction
    fragments: Tuple[Molecule, ...]
    num_reactions: int
    num_decompositions: int
    budget: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"SB({self.molecule}, {self.anchored_reaction}, {self.num_reactions}, {self.num_decompositions}, {self.budget})"


@dataclass(frozen=True)
class ReactionActionB:
    fragment: Molecule
    idx: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AB({self.idx})"


@dataclass(frozen=True)
class ReactionActionSpaceB(IndexedActionSpaceBase[ReactionActionB | ReactionActionExit]):
    possible_actions: Tuple[ReactionActionB, ...]
    exit_action: Optional[ReactionActionExit]

    def get_action_at_idx(self, idx: int) -> ReactionActionB | ReactionActionExit:
        if idx == len(self.possible_actions):
            assert self.exit_action is not None
            return self.exit_action
        else:
            return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionActionB | ReactionActionExit) -> int:
        if isinstance(action, ReactionActionExit):
            assert self.exit_action is not None
            return len(self.possible_actions)
        else:
            return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        extra = 1 if self.exit_action is not None else 0
        return list(range(len(self.possible_actions) + extra))

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_actions = self.possible_actions
        if self.exit_action is not None:
            possible_actions += (self.exit_action,)
        return f"ASB({possible_actions})"


@dataclass(frozen=True)
class ReactionStateC:
    molecule: Molecule
    anchored_reaction: AnchoredReaction
    fragments: Tuple[Molecule, ...]
    num_reactions: int
    num_decompositions: int
    budget: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"SC({self.molecule}, {self.anchored_reaction}, {self.fragments}, {self.num_reactions}, {self.num_decompositions}, {self.budget})"
        )


@dataclass(frozen=True)
class ReactionActionC:
    input_molecule: Molecule
    input_reaction: AnchoredReaction
    input_fragments: Tuple[Molecule, ...]
    output_molecule: Molecule

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AC({self.input_molecule}, {self.input_reaction}, {self.input_fragments}, {self.output_molecule})"


@dataclass(frozen=True)
class ReactionActionSpaceC(IndexedActionSpaceBase[ReactionActionC]):
    possible_actions: Tuple[ReactionActionC, ...]

    def get_action_at_idx(self, idx: int) -> ReactionActionC:
        return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionActionC) -> int:
        return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ASC({self.possible_actions})"


@dataclass(frozen=True)
class ReactionActionSpace0orCBackward(IndexedActionSpaceBase[ReactionAction0 | ReactionActionC | ReactionActionExit]):
    possible_actions: Tuple[ReactionAction0 | ReactionActionC, ...]
    exit_action: Optional[ReactionActionExit]

    def get_action_at_idx(self, idx: int) -> ReactionAction0 | ReactionActionC | ReactionActionExit:
        if idx == len(self.possible_actions):
            assert self.exit_action is not None
            return self.exit_action
        else:
            return self.possible_actions[idx]

    def get_idx_of_action(self, action: ReactionAction0 | ReactionActionC | ReactionActionExit) -> int:
        if isinstance(action, ReactionActionExit):
            assert self.exit_action is not None
            return len(self.possible_actions)
        else:
            return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        extra = 1 if self.exit_action is not None else 0
        return list(range(len(self.possible_actions) + extra))

    def __repr__(self):
        return str(self)

    def __str__(self):
        possible_actions = self.possible_actions
        if self.exit_action is not None:
            possible_actions += (self.exit_action,)
        return f"ASCBackward({possible_actions})"


@dataclass(frozen=True)
class ReactionStateEarlyTerminal:
    previous_state: Any = field(hash=False, compare=False)


@dataclass(frozen=True)
class ReactionActionEarlyTerminate:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ETA()"


@dataclass(frozen=True)
class ReactionActionSpaceEarlyTerminate(IndexedActionSpaceBase[ReactionActionEarlyTerminate]):
    possible_action = ReactionActionEarlyTerminate()

    def get_action_at_idx(self, idx: int) -> ReactionActionEarlyTerminate:
        return self.possible_action

    def get_idx_of_action(self, action: ReactionActionEarlyTerminate) -> int:
        return 0

    def get_possible_actions_indices(self) -> List[int]:
        return [0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ASETA()"


@dataclass(frozen=True)
class ReactionStateTerminal:
    molecule: Molecule = field(hash=True, compare=True)
    num_reactions: int = field(hash=False, compare=False)
    num_decompositions: int = field(hash=False, compare=False)
    budget: int

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ST({self.molecule}, {self.num_reactions}, {self.num_decompositions}, {self.budget})"


ReactionState = (
    ReactionState0
    | ReactionStateA
    | ReactionStateB
    | ReactionStateC
    | ReactionStateTerminal
    | ReactionStateEarlyTerminal
)
ReactionAction = ReactionAction0 | ReactionActionA | ReactionActionB | ReactionActionC
ReactionActionSpace = (
    ReactionActionSpace0
    | ReactionActionSpaceA
    | ReactionActionSpaceB
    | ReactionActionSpaceC
    | ReactionActionSpaceEarlyTerminate
)