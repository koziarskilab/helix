import abc
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Tuple, Type, Any

import gin
import torch
from torch import Tensor, nn
from torch.nn import Parameter, init

from rgfn.api.training_hooks_mixin import TrainingHooksMixin
from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.data_structures import Reaction, AnchoredReaction
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    Molecule,
    ReactionAction,
    ReactionActionC,
    ReactionActionSpace,
    ReactionActionSpace0Invalid,
    ReactionActionSpace0orCBackward,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceEarlyTerminate,
    ReactionState,
    ReactionStateC, ReactionActionSpaceX, ReactionActionSpaceY, ReactionActionSpaceHorZBackward,
    ReactionStateB, ReactionStateA, ReactionStateX, ReactionActionZ, ReactionStateTerminal, ReactionStateY,
    ReactionStateZ,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.policies.action_embeddings import ActionEmbeddingBase
from rgfn.gfns.reaction_gfn.policies.graph_transformer import (
    GraphTransformer,
    mol2graph,
    mols2batch,
)
from rgfn.gfns.reaction_gfn.policies.reaction_forward_policy import (
    ReactionForwardPolicy,
)
from rgfn.gfns.reaction_gfn.policies.utils import (
    OrderedSet,
    one_hot,
    to_dense_embeddings,
)
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace


@dataclass(frozen=True)
class SharedEmbeddings:
    molecule_to_idx: Dict[Molecule, int]
    molecule_reaction_to_idx: Dict[Tuple[Molecule, Reaction], int]

    all_embeddings: Tensor


class ReactantPositionalEncodingBase(abc.ABC, nn.Module, TrainingHooksMixin):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.data_factory = data_factory
        self._cache: Tensor | None = None
        self.device = "cpu"

    @abc.abstractmethod
    def _get_all_embeddings(self) -> Tensor:
        pass

    def get_all_embeddings(self) -> Tensor:
        if self._cache is None:
            self._cache = self._get_all_embeddings()
        return self._cache

    def clear_cache(self):
        self._cache = None

    def on_start_sampling(self, iteration_idx: int, recursive: bool = True) -> Dict[str, Any]:
        self._cache = None
        return {}

    def on_end_sampling(
        self,
        iteration_idx: int,
        trajectories_container: TrajectoriesContainer,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        self._cache = None
        return {}

    @abc.abstractmethod
    def select_embeddings(self, items: List[Tuple[AnchoredReaction, int]]) -> Tensor:
        pass


class ReactantSimplePositionalEncoding(ReactantPositionalEncodingBase):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int):
        super().__init__(data_factory, hidden_dim)
        self.anchored_reaction = data_factory.get_anchored_reactions()
        self.reaction_idx_to_embedding_idx = []
        total_embeddings = 0
        for reaction in self.anchored_reaction:
            self.reaction_idx_to_embedding_idx.append(total_embeddings)
            total_embeddings += len(reaction.fragment_patterns)

        self.weights = nn.Parameter(torch.empty(total_embeddings, hidden_dim), requires_grad=True)
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def _get_all_embeddings(self) -> Tensor:
        return self.weights

    def select_embeddings(self, items: List[Tuple[AnchoredReaction, int]]) -> Tensor:
        indices = [self.reaction_idx_to_embedding_idx[reaction.idx] + i for reaction, i in items]
        return torch.index_select(
            self.weights, index=torch.tensor(indices).long().to(self.device), dim=0
        )


@gin.configurable()
class ReactionBackwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, SharedEmbeddings],
):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        action_embedding_fn: ActionEmbeddingBase,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 5,
        linear_output: bool = False,
        backbone_policy: ReactionForwardPolicy | None = None,
    ):
        super().__init__()
        self.anchored_reactions = data_factory.get_anchored_reactions()
        self.reaction_to_idx = {
            reaction: idx for idx, reaction in enumerate(self.anchored_reactions)
        }
        self.use_backbone = backbone_policy is not None
        self.gnn = (
            GraphTransformer(
                x_dim=71,
                e_dim=4,
                g_dim=len(self.anchored_reactions),
                num_layers=num_layers,
                num_heads=num_heads,
                num_emb=hidden_dim,
            )
            if not self.use_backbone
            else backbone_policy.gnn
        )

        self.mlp_x = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if not linear_output
            else nn.Linear(hidden_dim, 1)
        )

        self.mlp_y = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if not linear_output
            else nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp_z = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if not linear_output
            else nn.Linear(hidden_dim, 1)
        )

        self.mlp_a = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(self.anchored_reactions) + 2),
            )
            if not linear_output
            else nn.Linear(hidden_dim, len(self.anchored_reactions) + 2)
        )

        self.mlp_b = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
            if not linear_output
            else nn.Linear(hidden_dim, 2)
        )
        self.mlp_b_fragments = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.b_reactant_pe = ReactantSimplePositionalEncoding(data_factory, hidden_dim)
        self.b_action_embedding_fn = action_embedding_fn

        self.mlp_c = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if not linear_output
            else nn.Linear(hidden_dim, 1)
        )

        self._action_space_type_to_forward_fn = {
            ReactionActionSpaceX: self._forward_x,
            ReactionActionSpaceY: self._forward_y,
            ReactionActionSpaceHorZBackward: self._forward_z,
            ReactionActionSpace0Invalid: self._forward_deterministic,
            ReactionActionSpaceA: self._forward_a,
            ReactionActionSpaceB: self._forward_b,
            ReactionActionSpace0orCBackward: self._forward_c,
            ReactionActionSpaceEarlyTerminate: self._forward_deterministic,
        }

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        return [self.b_action_embedding_fn, self.b_reactant_pe]

    @property
    def action_space_to_forward_fn(
        self,
    ) -> Dict[
        Type[TIndexedActionSpace],
        Callable[[List[TState], List[TIndexedActionSpace], TSharedEmbeddings], Tensor],
    ]:
        return self._action_space_type_to_forward_fn

    def _forward_x(
        self,
        states: List[ReactionStateY],
        action_spaces: List[ReactionActionSpaceX],
        shared_embeddings: SharedEmbeddings,
    ) -> Tensor:
        embedding_indices_list = []
        for action_space in action_spaces:
            embedding_indices = [
                shared_embeddings.molecule_to_idx[action.output_molecule]
                for action in action_space.possible_actions
            ]
            embedding_indices_list.append(embedding_indices)

        embedding_indices_flat = [idx for indices in embedding_indices_list for idx in indices]
        embedding_indices = torch.tensor(embedding_indices_flat).long().to(self.device)
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=embedding_indices, dim=0
        )
        logits = self.mlp_x(embeddings).squeeze(-1)
        logits, _ = to_dense_embeddings(
            logits, [len(indices) for indices in embedding_indices_list], fill_value=float("-inf")
        )
        return logits

    def _forward_y(
        self,
        states: List[ReactionStateZ],
        action_spaces: List[ReactionActionSpaceY],
        shared_embeddings: SharedEmbeddings,
    ) -> Tensor:
        mol_embedding_indices = [
            shared_embeddings.molecule_reaction_to_idx[(state.molecule, state.anchored_reaction)]
            for state in states
        ]
        mol_embedding_indices = torch.tensor(mol_embedding_indices).long().to(self.device)
        mol_embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=mol_embedding_indices, dim=0
        )

        actions_indices = [
            [action.idx for action in action_space.possible_actions]
            for action_space in action_spaces
        ]
        actions_indices_flat = [idx for indices in actions_indices for idx in indices]
        action_indices_flat = torch.tensor(actions_indices_flat).long().to(self.device)
        all_action_embeddings_flat = self.b_action_embedding_fn.get_embeddings()
        action_embeddings = torch.index_select(
            all_action_embeddings_flat, index=action_indices_flat, dim=0
        )
        actions_embeddings, mask = to_dense_embeddings(
            action_embeddings, [len(indices) for indices in actions_indices], fill_value=0
        )  # (batch_size, max_num_actions, hidden_dim)

        # Augmenting mol embedding with chosen fragments embeddings
        augmented_indices = []
        fragment_indices = []
        positional_encodings_indices = []
        for state_idx, state in enumerate(states):
            for positional_idx, fragment in enumerate(state.fragments):
                augmented_indices.append(state_idx)
                fragment_indices.append(fragment.idx)
                positional_encodings_indices.append((state.anchored_reaction, positional_idx))

        if len(augmented_indices) > 0:
            positional_encodings = self.b_reactant_pe.select_embeddings(
                positional_encodings_indices
            )
            fragment_indices = torch.tensor(fragment_indices).long().to(self.device)
            fragment_embeddings = torch.index_select(
                all_action_embeddings_flat, 0, fragment_indices
            )
            fragment_embeddings = fragment_embeddings + positional_encodings
            fragment_embeddings = self.mlp_b_fragments(fragment_embeddings)

            augmented_indices = torch.tensor(augmented_indices).long().to(self.device)
            mol_embeddings = torch.index_add(
                mol_embeddings, 0, augmented_indices, fragment_embeddings
            )

        mol_embeddings = self.mlp_y(mol_embeddings)  # (batch_size, hidden_dim)
        logits = torch.matmul(actions_embeddings, mol_embeddings.unsqueeze(2)).squeeze(
            2
        )  # (batch_size, max_num_actions)
        return torch.masked_fill(logits, ~mask, float("-inf"))

    def _forward_z(
        self,
        states: List[ReactionStateX],
        action_spaces: List[ReactionActionSpaceHorZBackward],
        shared_embeddings: SharedEmbeddings,
    ) -> Tensor:
        embedding_indices_list = []
        for state, action_space in zip(states, action_spaces):
            embedding_indices = []
            for action in action_space.possible_actions:
                if isinstance(action, ReactionActionZ):
                    embedding_indices.append(
                        shared_embeddings.molecule_reaction_to_idx[
                            (state.molecule, action.anchored_reaction)
                        ]
                    )
                else:
                    embedding_indices.append(
                        shared_embeddings.molecule_reaction_to_idx[(action.fragment, None)]
                    )
            embedding_indices_list.append(embedding_indices)

        embedding_indices_flat = [idx for indices in embedding_indices_list for idx in indices]
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings,
            index=torch.tensor(embedding_indices_flat).long().to(self.device),
            dim=0
        )
        logits = self.mlp_z(embeddings).squeeze(-1)
        logits, _ = to_dense_embeddings(
            logits, [len(indices) for indices in embedding_indices_list], fill_value=float("-inf")
        )
        return logits

    def _forward_a(
        self,
        states: List[ReactionStateB | ReactionStateC | ReactionStateTerminal],
        action_spaces: List[ReactionActionSpaceA],
        shared_embeddings: SharedEmbeddings,
    ) -> Tensor:
        embedding_indices = [
            shared_embeddings.molecule_to_idx[state.molecule]
            if isinstance(state, ReactionStateTerminal) else
            shared_embeddings.molecule_reaction_to_idx[(state.molecule, state.anchored_reaction)]
            for state in states
        ]
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings,
            index=torch.tensor(embedding_indices).long().to(self.device),
            dim=0
        )
        logits = self.mlp_a(embeddings)
        mask = torch.tensor([
            action_space.possible_actions_mask + [1 if action_space.exit_action else 0]
            for action_space in action_spaces]
        ).bool().to(self.device)
        logits = torch.masked_fill(logits, ~mask, float("-inf"))
        return logits

    def _forward_b(
        self,
        states: List[ReactionStateB | ReactionStateC],
        action_spaces: List[ReactionActionSpaceB],
        shared_embeddings: SharedEmbeddings,
    ) -> Tensor:
        mol_embedding_indices = [
            shared_embeddings.molecule_reaction_to_idx[(state.molecule, state.anchored_reaction)]
            for state in states
        ]
        mol_embeddings = torch.index_select(
            shared_embeddings.all_embeddings,
            index=torch.tensor(mol_embedding_indices).long().to(self.device),
            dim=0
        )

        # Augmenting mol embedding with chosen fragments embeddings
        augmented_indices = []
        fragment_indices = []
        positional_encodings_indices = []
        for state_idx, state in enumerate(states):
            for positional_idx, fragment in enumerate(state.fragments):
                augmented_indices.append(state_idx)
                fragment_indices.append(fragment.idx)
                positional_encodings_indices.append((state.anchored_reaction, positional_idx))

        if len(augmented_indices) > 0:
            positional_encodings = self.b_reactant_pe.select_embeddings(
                positional_encodings_indices
            )
            all_action_embeddings_flat = self.b_action_embedding_fn.get_embeddings()
            fragment_indices = torch.tensor(fragment_indices).long().to(self.device)
            fragment_embeddings = torch.index_select(
                all_action_embeddings_flat, 0, fragment_indices
            )
            fragment_embeddings = fragment_embeddings + positional_encodings
            fragment_embeddings = self.mlp_b_fragments(fragment_embeddings)

            augmented_indices = torch.tensor(augmented_indices).long().to(self.device)
            mol_embeddings = torch.index_add(
                mol_embeddings, 0, augmented_indices, fragment_embeddings
            )

        logits = self.mlp_b(mol_embeddings)
        mask = torch.tensor([
            [True, action_space.exit_action is not None]
            for action_space in action_spaces]
        ).to(self.device)
        logits = torch.masked_fill(logits, ~mask, float("-inf"))
        return logits

    def _forward_c(
        self,
        states: List[ReactionStateA],
        action_spaces: List[ReactionActionSpace0orCBackward],
        shared_embeddings: SharedEmbeddings,
    ) -> Tensor:
        embedding_indices_list = []
        for state, action_space in zip(states, action_spaces):
            embedding_indices = []
            for action in action_space.possible_actions:
                if isinstance(action, ReactionActionC):
                    embedding_indices.append(
                        shared_embeddings.molecule_reaction_to_idx[
                            (action.input_molecule, action.input_reaction)
                        ]
                    )
                else:
                    embedding_indices.append(
                        shared_embeddings.molecule_reaction_to_idx[(action.fragment, None)]
                    )
            # exit action is actually always available if not using recurrence
            if action_space.exit_action is not None:
                embedding_indices.append(
                    shared_embeddings.molecule_to_idx[state.molecule]
                )
            embedding_indices_list.append(embedding_indices)

        embedding_indices_flat = [idx for indices in embedding_indices_list for idx in indices]
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings,
            index=torch.tensor(embedding_indices_flat).long().to(self.device),
            dim=0
        )
        logits = self.mlp_c(embeddings).squeeze(-1)
        logits, _ = to_dense_embeddings(
            logits, [len(indices) for indices in embedding_indices_list], fill_value=float("-inf")
        )
        return logits

    def _forward_deterministic(
        self,
        states: List[ReactionState],
        action_spaces: List[ReactionActionSpace],
        shared_embeddings: SharedEmbeddings,
    ) -> Tensor:
        assert len(states) == len(action_spaces)
        max_action_idx = max(
            action_space.get_possible_actions_indices()[0] for action_space in action_spaces
        )
        logits_list = []
        for action_space in action_spaces:
            logits = [-float("inf")] * (max_action_idx + 1)
            logits[action_space.get_possible_actions_indices()[0]] = 0
            logits_list.append(logits)
        return torch.tensor(logits_list).float().to(self.device)

    def get_shared_embeddings(
        self, states: List[ReactionState], action_spaces: List[ReactionActionSpace]
    ) -> SharedEmbeddings:
        all_molecules = OrderedSet()
        all_molecules_reactions = OrderedSet()
        for state, action_space in zip(states, action_spaces):
            if isinstance(action_space, ReactionActionSpaceA):
                if isinstance(state, ReactionStateTerminal):
                    all_molecules.add(state.molecule)
                else:
                    all_molecules_reactions.add((state.molecule, state.anchored_reaction))
            elif isinstance(action_space, ReactionActionSpaceB):
                all_molecules_reactions.add((state.molecule, state.anchored_reaction))
            elif isinstance(action_space, ReactionActionSpace0orCBackward):
                for action in action_space.possible_actions:
                    if isinstance(action, ReactionActionC):
                        all_molecules_reactions.add((action.input_molecule, action.input_reaction))
                    else:
                        all_molecules_reactions.add((action.fragment, None))
                if action_space.exit_action is not None:
                    all_molecules.add(state.molecule)
            elif isinstance(action_space, ReactionActionSpaceX):
                for action in action_space.possible_actions:
                    all_molecules.add(action.output_molecule)
            elif isinstance(action_space, ReactionActionSpaceY):
                all_molecules_reactions.add((state.molecule, state.anchored_reaction))
            elif isinstance(action_space, ReactionActionSpaceHorZBackward):
                for action in action_space.possible_actions:
                    if isinstance(action, ReactionActionZ):
                        all_molecules_reactions.add((state.molecule, action.anchored_reaction))
                    else:
                        all_molecules_reactions.add((action.fragment, None))

        molecule_to_idx = {molecule: idx for idx, molecule in enumerate(all_molecules)}
        molecule_and_reaction_to_idx = {
            molecule_reaction: idx + len(molecule_to_idx)
            for idx, molecule_reaction in enumerate(all_molecules_reactions)
        }

        molecule_graphs = [
            mol2graph(mol.rdkit_mol if mol else None) for mol in molecule_to_idx.keys()
        ]
        reaction_cond = [one_hot(0, len(self.anchored_reactions))] * len(molecule_to_idx)

        molecule_and_reaction_graphs = [
            mol2graph(mol.rdkit_mol) for mol, _ in molecule_and_reaction_to_idx.keys()
        ]
        molecule_and_reaction_cond = [
            one_hot(r.idx, len(self.anchored_reactions))
            if r is not None
            else one_hot(-1, len(self.anchored_reactions))
            for _, r in molecule_and_reaction_to_idx.keys()
        ]

        graphs = molecule_graphs + molecule_and_reaction_graphs
        conds = reaction_cond + molecule_and_reaction_cond
        if len(graphs) > 0:
            graph_batch = mols2batch(graphs).to(self.device)
            cond_batch = torch.tensor(conds).float().to(self.device)
            embeddings = self.gnn(graph_batch, cond_batch)
        else:
            embeddings = None
        return SharedEmbeddings(
            molecule_to_idx=molecule_to_idx,
            molecule_reaction_to_idx=molecule_and_reaction_to_idx,
            all_embeddings=embeddings,
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if not self.use_backbone:
            return super().parameters(recurse)
        return self.mlp_c.parameters(recurse)

    def compute_states_log_flow(self, states: List[ReactionState]) -> Tensor:
        raise NotImplementedError()
