import itertools
from typing import Any, Dict, List, Tuple, Set, Optional

import gin
from rdkit import RDLogger
from rdkit.Chem import Mol

from rgfn.api.env_base import EnvBase
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.data_structures import Cache, Pattern
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    Molecule,
    ReactionAction,
    ReactionAction0,
    ReactionAction0Invalid,
    ReactionActionA,
    ReactionActionB,
    ReactionActionC,
    ReactionActionEarlyTerminate,
    ReactionActionSpace,
    ReactionActionSpace0,
    ReactionActionSpace0Invalid,
    ReactionActionSpace0orCBackward,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceC,
    ReactionActionSpaceEarlyTerminate,
    ReactionState,
    ReactionState0,
    ReactionState0Invalid,
    ReactionStateA,
    ReactionStateB,
    ReactionStateC,
    ReactionStateEarlyTerminal,
    ReactionStateTerminal, ReactionStateH, ReactionActionSpaceH, ReactionActionH, ReactionStateX,
    ReactionActionX, ReactionActionExit, ReactionActionSpaceX, ReactionStateY, ReactionActionY, ReactionActionSpaceY,
    ReactionStateZ, ReactionActionSpaceZ, ReactionActionZ, ReactionActionSpaceHorZBackward
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory

RDLogger.DisableLog("rdApp.*")


@gin.configurable()
class ReactionEnv(EnvBase[ReactionState, ReactionActionSpace, ReactionAction]):
    def reversed(self) -> "EnvBase[ReactionState, ReactionActionSpace, ReactionAction]":
        env = super().reversed()
        if self.share_cache:
            env.action_space_cache = self.action_space_cache
            env.recurrence_cache = self.recurrence_cache
        return env

    def on_update_fragments_library(
        self,
        iteration_idx: int,
        fragments: List[Molecule],
        costs: List[float],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        new_fragments = self.data_factory.get_fragments() + fragments
        assert all(idx == f.idx for idx, f in enumerate(new_fragments))
        self._setup_fragments(new_fragments)
        self.action_space_cache.clear()
        self.recurrence_cache.clear()
        return {}

    def __init__(
        self,
        data_factory: ReactionDataFactory,
        max_num_reactions: int,
        max_num_decompositions: Optional[int] = None,
        starting_smiles_with_budget: Optional[List[Tuple[str, int]]] = None,
        use_recurrence: bool = False,
        use_cache: bool = True,
        share_cache: bool = True,
    ):
        super().__init__()
        self.data_factory = data_factory
        self.reactions = data_factory.get_reactions()
        self.disconnections = data_factory.get_disconnections()
        self.anchored_reactions = data_factory.get_anchored_reactions()
        self.reaction_anchor_map = data_factory.get_reaction_anchor_map()
        self.anchored_disconnections = data_factory.get_anchored_disconnections()
        self.max_num_reaction = max_num_reactions
        self.max_num_decompositions = max_num_decompositions
        self.action_space_cache = Cache(max_size=50_000)
        self.recurrence_cache = Cache(max_size=10_000_000)
        self.use_cache = use_cache
        self.share_cache = share_cache
        self.use_recurrence = use_recurrence

        self.composable_cache = Cache(max_size=10_000_000)
        self.future_composable_cache = Cache(max_size=50_000)

        if starting_smiles_with_budget is None:
            starting_smiles_with_budget = []
            assert max_num_decompositions is None
        else:
            assert max_num_decompositions is not None
        starting_molecules_with_budget = [
            (Molecule(smiles), budget)
            for (smiles, budget) in starting_smiles_with_budget
        ]

        self.all_actions_a = tuple(
            ReactionActionA(anchored_reaction=r, idx=i)
            for i, r in enumerate(self.anchored_reactions)
        ) + (
            ReactionActionA(anchored_reaction=None, idx=len(self.anchored_reactions)),
        )

        self.all_actions_z = tuple(
            ReactionActionZ(anchored_reaction=r, idx=i)
            for i, r in enumerate(self.anchored_reactions)
        )
        # we don't need `ReactionActionA(anchored_reaction=None, idx=len(self.anchored_reactions))` here
        # `exit_action` is pretty much the same thing, but in an explicit way

        self.fragments: List[Molecule] = ...
        self.all_actions_0: Tuple[ReactionAction0, ...] = ...
        self.all_actions_b: Tuple[ReactionActionB, ...] = ...
        self.pattern_to_compatible_fragments: Dict[Pattern, List[Molecule]] = ...
        self.pattern_to_min_fragment_num_reactions: Dict[Pattern, int] = ...
        self.smiles_to_fragment_idx: Dict[str, int] = ...
        self.fragment_smiles_set: Set[str] = ...

        self._setup_fragments(fragments=data_factory.get_fragments())

        self.starting_molecules_with_budget: List[Tuple[Molecule, int]] = ...
        self.all_actions_h: Tuple[ReactionActionH, ...] = ...
        self.all_actions_y: Tuple[ReactionActionY, ...] = ...
        self.starting_molecules_smiles_set: Set[str] = ...
        self.smiles_to_action_h_idx: Dict[str, int] = ...
        self.max_initial_budget: int = ...

        self._setup_starting_molecules(fragments_with_budget=starting_molecules_with_budget)

        self.forward_action_space_dict = {
            ReactionStateH: self._get_forward_action_spaces_h,
            ReactionStateX: self._get_forward_action_spaces_x,
            ReactionStateY: self._get_forward_action_spaces_y,
            ReactionStateZ: self._get_forward_action_spaces_z,
            ReactionState0: self._get_forward_action_spaces_0,
            ReactionState0Invalid: self._get_forward_action_spaces_0_invalid,
            ReactionStateA: self._get_forward_action_spaces_a,
            ReactionStateB: self._get_forward_action_spaces_b,
            ReactionStateC: self._get_forward_action_spaces_c,
        }
        self.backward_action_space_dict = {
            ReactionStateX: self._get_backward_action_spaces_x,
            ReactionStateY: self._get_backward_action_spaces_y,
            ReactionStateZ: self._get_backward_action_spaces_z,
            ReactionStateA: self._get_backward_action_spaces_a,
            ReactionStateB: self._get_backward_action_spaces_b,
            ReactionStateC: self._get_backward_action_spaces_c,
            ReactionStateTerminal: self._get_backward_action_spaces_terminal,
            ReactionStateEarlyTerminal: self._get_backward_action_spaces_early_terminal,
        }
        self.forward_action_dict = {
            ReactionActionH: self._apply_forward_actions_h,
            ReactionActionX: self._apply_forward_actions_x,
            ReactionActionY: self._apply_forward_actions_y,
            ReactionActionZ: self._apply_forward_actions_z,
            ReactionActionExit: self._apply_forward_actions_exit,
            ReactionAction0: self._apply_forward_actions_0,
            ReactionAction0Invalid: self._apply_forward_actions_0_invalid,
            ReactionActionA: self._apply_forward_actions_a,
            ReactionActionB: self._apply_forward_actions_b,
            ReactionActionC: self._apply_forward_actions_c,
            ReactionActionEarlyTerminate: self._apply_forward_actions_early_terminate,
        }
        self.backward_action_dict = {
            ReactionActionH: self._apply_backward_actions_h,
            ReactionActionX: self._apply_backward_actions_x,
            ReactionActionY: self._apply_backward_actions_y,
            ReactionActionZ: self._apply_backward_actions_z,
            ReactionActionExit: self._apply_backward_actions_exit,
            ReactionAction0: self._apply_backward_actions_0,
            ReactionAction0Invalid: self._apply_backward_actions_0_invalid,
            ReactionActionA: self._apply_backward_actions_a,
            ReactionActionB: self._apply_backward_actions_b,
            ReactionActionC: self._apply_backward_actions_c,
            ReactionActionEarlyTerminate: self._apply_backward_actions_early_terminate,
        }

    def _setup_fragments(self, fragments: List[Molecule]):
        self.fragments = fragments
        assert all(idx == f.idx for idx, f in enumerate(self.fragments))
        self.all_actions_0 = tuple(
            ReactionAction0(fragment=f, idx=i) for i, f in enumerate(self.fragments)
        )

        self.all_actions_b = tuple(
            ReactionActionB(fragment=f, idx=i) for i, f in enumerate(self.fragments)
        )

        self.pattern_to_compatible_fragments: Dict[Pattern, List[Molecule]] = {}
        self.pattern_to_min_fragment_num_reactions: Dict[Pattern, int] = {}
        for reaction in self.reactions:
            for pattern in reaction.left_side_patterns:
                self.pattern_to_min_fragment_num_reactions[pattern] = self.max_num_reaction + 10
                fragments = []
                for i, fragment in enumerate(self.fragments):
                    if fragment.rdkit_mol.HasSubstructMatch(pattern.rdkit_pattern):
                        fragments.append(fragment)
                        self.pattern_to_min_fragment_num_reactions[pattern] = min(
                            self.pattern_to_min_fragment_num_reactions[pattern],
                            fragment.num_reactions,
                        )
                self.pattern_to_compatible_fragments[pattern] = fragments

        self.smiles_to_fragment_idx = {fragment.smiles: fragment.idx for fragment in self.fragments}
        self.fragment_smiles_set = set(self.smiles_to_fragment_idx.keys())

    def _setup_starting_molecules(self, fragments_with_budget: List[Tuple[Molecule, int]]):
        self.starting_molecules_with_budget = fragments_with_budget
        self.starting_molecules_smiles_set = set()
        self.smiles_to_action_h_idx = dict()

        self.all_actions_h = tuple(
            ReactionActionH(fragment=f, budget=b, idx=i)
            for i, (f, b) in enumerate(self.starting_molecules_with_budget)
        )

        self.all_actions_y = tuple(
            ReactionActionY(fragment=f, idx=i) for i, f in enumerate(self.fragments)
        )

        for idx, (fragment, budget) in enumerate(self.starting_molecules_with_budget):
            self.starting_molecules_smiles_set.add(fragment.smiles)
            self.smiles_to_action_h_idx[fragment.smiles] = idx

        self.max_initial_budget = max(budget for _, budget in self.starting_molecules_with_budget)

        # ensure no duplicate (fragments, budget) pairs
        assert len(self.starting_molecules_with_budget) == len(self.smiles_to_action_h_idx)

    def get_forward_action_spaces(self, states: List[ReactionState]) -> List[ReactionActionSpace]:
        action_spaces = []
        for state in states:
            action_space = self.forward_action_space_dict[type(state)](state)
            action_spaces.append(action_space)
        return action_spaces

    def _get_forward_action_spaces_h(self, state: ReactionStateH) -> ReactionActionSpaceH:
        return ReactionActionSpaceH(
            all_actions=self.all_actions_h, possible_actions_mask=[True] * len(self.all_actions_h)
        )

    def _get_forward_action_spaces_x(
        self, state: ReactionStateX
    ) -> ReactionActionSpaceX:
        if state.num_decompositions >= self.max_num_decompositions:
            return ReactionActionSpaceX(possible_actions=(), exit_action=ReactionActionExit())

        possible_actions: List[ReactionActionX] = []
        possible_actions_set = set()

        for reaction, disconnection in zip(self.reactions, self.disconnections):
            reactants_list_list = disconnection.rdkit_rxn.RunReactants(
                [state.molecule.rdkit_mol]
            )
            for reactants_list in reactants_list_list:
                fragments, non_fragments, non_fragment_indices = self._lazy_is_fragment_check(
                    reactants_list
                )
                if len(non_fragments) > 1:
                    continue

                if len(non_fragments) == 0:
                    for idx, previous_molecule in enumerate(fragments):
                        previous_fragments = fragments[:idx] + fragments[idx + 1:]
                        action = ReactionActionX(
                            input_molecule=previous_molecule,
                            input_reaction=self.reaction_anchor_map[(reaction, idx)],
                            input_fragments=tuple(previous_fragments),
                            output_molecule=state.molecule,
                        )
                        if (
                            action not in possible_actions_set
                            and self._is_action_product_selection_valid_in_forward(action)
                        ):
                            possible_actions.append(action)
                            possible_actions_set.add(action)
                else:
                    previous_molecule = non_fragments[0]
                    if not previous_molecule.valid:
                        continue
                    action = ReactionActionX(
                        input_molecule=previous_molecule,
                        input_reaction=self.reaction_anchor_map[
                            (reaction, non_fragment_indices[0])
                        ],
                        input_fragments=tuple(fragments),
                        output_molecule=state.molecule,
                    )
                    if (
                        action not in possible_actions_set
                        and self._is_action_product_selection_valid_in_forward(action)
                    ):
                        possible_actions.append(action)
                        possible_actions_set.add(action)

        exit_action = None
        if state.budget > 0:
            exit_action = ReactionActionExit()

        return ReactionActionSpaceX(
            possible_actions=tuple(possible_actions),
            exit_action=exit_action
        )

    def _get_forward_action_spaces_y(
        self, state: ReactionStateY
    ) -> ReactionActionSpaceY | ReactionActionSpaceZ:
        exit_action = None
        if state.budget > 0:
            exit_action = ReactionActionExit()

        if len(state.fragments) == 0:
            return ReactionActionSpaceZ(
                possible_action=self.all_actions_z[state.anchored_reaction.idx],
                exit_action=exit_action,
            )
        previous_fragment_smiles = state.fragments[-1].smiles
        previous_fragment = self.fragments[self.smiles_to_fragment_idx[previous_fragment_smiles]]
        return ReactionActionSpaceY(
            possible_actions=(self.all_actions_y[previous_fragment.idx],),
            exit_action=exit_action,
        )

    def _get_forward_action_spaces_z(
        self, state: ReactionStateZ
    ) -> ReactionActionSpaceY | ReactionActionSpaceZ:
        state_y = ReactionStateY(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            fragments=state.fragments,
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )
        return self._get_forward_action_spaces_y(state_y)


    def _get_forward_action_spaces_0(self, state: ReactionState0) -> ReactionActionSpace0:
        return ReactionActionSpace0(
            all_actions=self.all_actions_0, possible_actions_mask=[True] * len(self.all_actions_0)
        )

    def _get_forward_action_spaces_0_invalid(
        self, state: ReactionState0Invalid
    ) -> ReactionActionSpace0Invalid:
        return ReactionActionSpace0Invalid()

    def _get_forward_action_spaces_a(
        self, state: ReactionStateA
    ) -> ReactionActionSpaceA | ReactionActionSpaceEarlyTerminate:
        mask = [False] * len(self.all_actions_a)

        # Allow for exit action if state is non-empty
        if state.num_reactions > 0 and state.molecule.smiles not in self.starting_molecules_smiles_set:
            mask[-1] = True

        if state.budget > 0:
            mol = state.molecule.rdkit_mol
            for i, action in enumerate(self.all_actions_a[:-1]):
                action: ReactionActionA
                anchored_reaction = action.anchored_reaction
                anchored_pattern = anchored_reaction.anchored_pattern
                if not mol.HasSubstructMatch(anchored_pattern.rdkit_pattern):
                    continue
                # sum_min_num_reactions = sum(
                #     self.pattern_to_min_fragment_num_reactions[pattern]
                #     for pattern in anchored_reaction.fragment_patterns
                # )
                # if sum_min_num_reactions <= self.max_num_reaction - state.num_reactions:
                mask[i] = True

        if not any(mask):
            return ReactionActionSpaceEarlyTerminate()
        return ReactionActionSpaceA(
            all_actions=self.all_actions_a,
            possible_actions_mask=mask,
            exit_action=None
        )

    def _get_forward_action_spaces_b(
        self, state: ReactionStateB
    ) -> ReactionActionSpaceB | ReactionActionSpaceEarlyTerminate:
        next_pattern = state.anchored_reaction.fragment_patterns[len(state.fragments)]
        # rest_min_sum_num_reactions = sum(
        #     self.pattern_to_min_fragment_num_reactions[pattern]
        #     for pattern in state.anchored_reaction.fragment_patterns[len(state.fragments) + 1 :]
        # )
        # max_num_reactions = self.max_num_reaction - state.num_reactions - rest_min_sum_num_reactions
        possible_actions = [
            self.all_actions_b[fragment.idx]
            for fragment in self.pattern_to_compatible_fragments[next_pattern]
            # if fragment.num_reactions < max_num_reactions
        ]

        if len(possible_actions) == 0:
            return ReactionActionSpaceEarlyTerminate()

        return ReactionActionSpaceB(
            possible_actions=tuple(possible_actions),
            exit_action=None
        )

    def _get_forward_action_spaces_c(
        self, state: ReactionStateC
    ) -> ReactionActionSpaceC | ReactionActionSpaceEarlyTerminate:
        anchored_reaction = state.anchored_reaction
        reactants = [state.molecule] + list(state.fragments)
        products = anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])

        products_list = [Molecule(mol[0]) for mol in products]
        products_list = [mol for mol in products_list if mol.valid]

        anchored_disconnection = self.anchored_disconnections[anchored_reaction.idx]
        expected_reactants = tuple(reactants)
        possible_actions = []
        for product in dict.fromkeys(products_list):  # remove duplicates in deterministic manner
            prev_reactants_list = anchored_disconnection.rdkit_rxn.RunReactants([product.rdkit_mol])
            for prev_reactants in prev_reactants_list:
                prev_reactants = tuple(Molecule(prev) for prev in prev_reactants)
                if prev_reactants == expected_reactants:
                    action = ReactionActionC(
                        input_molecule=state.molecule,
                        input_reaction=state.anchored_reaction,
                        input_fragments=state.fragments,
                        output_molecule=product,
                    )
                    possible_actions.append(action)
                    if self.use_recurrence and not self.is_reversed:
                        self.recurrence_cache[(product.smiles, state.num_reactions + 1)] = True
                    break

        if len(possible_actions) == 0:
            return ReactionActionSpaceEarlyTerminate()
        return ReactionActionSpaceC(possible_actions=tuple(possible_actions))

    def get_backward_action_spaces(self, states: List[ReactionState]) -> List[ReactionActionSpace]:
        action_spaces = []
        for state in states:
            action_space = self.backward_action_space_dict[type(state)](state)
            action_spaces.append(action_space)
        return action_spaces

    def _is_action_product_selection_valid_in_forward(self, action: ReactionActionC | ReactionActionX):
        reactants_rdkit = [action.input_molecule.rdkit_mol] + [
            r.rdkit_mol for r in action.input_fragments
        ]
        new_products = action.input_reaction.rdkit_rxn.RunReactants(reactants_rdkit)
        for mol in new_products:
            product = Molecule(mol[0])
            if product.valid and action.output_molecule == product:
                return True
        return False

    def _get_backward_action_spaces_x(
        self, state: ReactionStateX
    ) -> ReactionActionSpaceHorZBackward | ReactionActionSpace0Invalid:
        possible_actions: List[ReactionActionH | ReactionActionZ] = []
        if state.molecule.smiles in self.starting_molecules_smiles_set:
            action_h_idx = self.smiles_to_action_h_idx[state.molecule.smiles]
            action_h = self.all_actions_h[action_h_idx]
            if state.budget <= action_h.budget:
                possible_actions.append(action_h)

        if state.budget > 0:
            for anchored_reaction in self.anchored_reactions:
                anchored_pattern = anchored_reaction.anchored_pattern
                if not state.molecule.rdkit_mol.HasSubstructMatch(anchored_pattern.rdkit_pattern):
                    continue

                fragment_patterns = anchored_reaction.fragment_patterns
                fragments_product = (
                    self.pattern_to_compatible_fragments[pattern] for pattern in fragment_patterns
                )
                found = False
                for fragments in itertools.product(*fragments_product):
                    reactants = (state.molecule,) + fragments
                    products = anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])
                    products_list = (Molecule(mol[0]) for mol in products)
                    products_list = (mol for mol in products_list if mol.valid)
                    for product in products_list:
                        if not self.use_recurrence or self._is_composable(product, state.num_decompositions - 1):
                            action_z = ReactionActionZ(
                                anchored_reaction=anchored_reaction,
                                idx=anchored_reaction.idx
                            )
                            possible_actions.append(action_z)
                            found = True
                            break
                    if found:
                        break

        if len(possible_actions) == 0:
            return ReactionActionSpace0Invalid()
        return ReactionActionSpaceHorZBackward(possible_actions=tuple(possible_actions))

    def _get_backward_action_spaces_y(
        self, state: ReactionStateY
    ) -> ReactionActionSpaceX | ReactionActionSpace0Invalid:
        if state.budget <= 0:
            return ReactionActionSpace0Invalid()

        anchored_reaction = state.anchored_reaction
        reactants = [state.molecule] + list(state.fragments)
        products = anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])

        products_list = [Molecule(mol[0]) for mol in products]
        products_list = [mol for mol in products_list if mol.valid]

        anchored_disconnection = self.anchored_disconnections[anchored_reaction.idx]
        expected_reactants = tuple(reactants)
        possible_actions: List[ReactionActionX] = []
        for product in dict.fromkeys(products_list):
            prev_reactants_list = anchored_disconnection.rdkit_rxn.RunReactants([product.rdkit_mol])
            for prev_reactants in prev_reactants_list:
                prev_reactants = tuple(Molecule(prev) for prev in prev_reactants)
                if prev_reactants == expected_reactants:
                    action = ReactionActionX(
                        input_molecule=state.molecule,
                        input_reaction=state.anchored_reaction,
                        input_fragments=state.fragments,
                        output_molecule=product,
                    )
                    possible_actions.append(action)
                    break

        if len(possible_actions) == 0:
            return ReactionActionSpace0Invalid()
        return ReactionActionSpaceX(possible_actions=tuple(possible_actions), exit_action=None)

    def _get_backward_action_spaces_z(
        self, state: ReactionStateZ
    ) -> ReactionActionSpaceY | ReactionActionSpace0Invalid:
        if state.budget <= 0:
            return ReactionActionSpace0Invalid()

        possible_actions: List[ReactionActionY] = []

        if not self.use_recurrence:
            next_pattern = state.anchored_reaction.fragment_patterns[len(state.fragments)]
            possible_actions = [
                self.all_actions_y[fragment.idx]
                for fragment in self.pattern_to_compatible_fragments[next_pattern]
            ]
        else:
            unfilled_patterns = state.anchored_reaction.fragment_patterns[len(state.fragments):]
            unfilled_fragments_product = (
                self.pattern_to_compatible_fragments[pattern] for pattern in unfilled_patterns
            )
            possible_fragment_set: Set[Molecule] = set()
            for unfilled_fragments in itertools.product(*unfilled_fragments_product):
                fragment: Molecule = unfilled_fragments[0]
                if fragment in possible_fragment_set:
                    continue
                reactants = (state.molecule,) + state.fragments + unfilled_fragments
                products = state.anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])
                products_list = (Molecule(mol[0]) for mol in products)
                products_list = (mol for mol in products_list if mol.valid)
                for product in products_list:
                    if self._is_composable(product, state.num_decompositions - 1):
                        action_y = ReactionActionY(
                            fragment=fragment,
                            idx=fragment.idx,
                        )
                        possible_actions.append(action_y)
                        possible_fragment_set.add(fragment)

        if len(possible_actions) == 0:
            return ReactionActionSpace0Invalid()
        return ReactionActionSpaceY(possible_actions=tuple(possible_actions), exit_action=None)

    def _get_backward_action_spaces_a(
        self, state: ReactionStateA
    ) -> ReactionActionSpace0orCBackward | ReactionActionSpace0Invalid:
        if self.use_cache and state in self.action_space_cache:
            return self.action_space_cache[state]

        if self.starting_molecules_smiles_set:
            action_space = self._get_backward_action_spaces_a_optimization(state)
        else:
            action_space = self._get_backward_action_spaces_a_standard(state)

        if self.use_cache:
            self.action_space_cache[state] = action_space
        return action_space

    def _get_backward_action_spaces_a_optimization(
        self, state: ReactionStateA
    ) -> ReactionActionSpace0orCBackward | ReactionActionSpace0Invalid:
        possible_actions: List[ReactionActionC] = []
        possible_actions_set = set()

        if state.budget < self.max_num_decompositions + self.max_initial_budget and state.num_reactions > 0:
            for reaction, disconnection in zip(self.reactions, self.disconnections):
                reactants_list_list = disconnection.rdkit_rxn.RunReactants(
                    [state.molecule.rdkit_mol]
                )
                for reactants_list in reactants_list_list:
                    fragments, non_fragments, non_fragment_indices = self._lazy_is_fragment_check(
                        reactants_list
                    )
                    if len(non_fragments) > 1:
                        continue

                    if len(non_fragments) == 0:
                        for idx, previous_molecule in enumerate(fragments):
                            previous_fragments = fragments[:idx] + fragments[idx + 1:]
                            action = ReactionActionC(
                                input_molecule=previous_molecule,
                                input_reaction=self.reaction_anchor_map[(reaction, idx)],
                                input_fragments=tuple(previous_fragments),
                                output_molecule=state.molecule,
                            )
                            if (
                                action not in possible_actions_set
                                and self._is_action_product_selection_valid_in_forward(action)
                                and (
                                    not self.use_recurrence
                                    or self._is_future_composable(
                                        previous_molecule,
                                        state.num_reactions - 1,
                                        state.num_decompositions,
                                    )
                                )
                            ):
                                possible_actions.append(action)
                                possible_actions_set.add(action)
                    else:
                        previous_molecule = non_fragments[0]
                        if not previous_molecule.valid:
                            continue
                        action = ReactionActionC(
                            input_molecule=previous_molecule,
                            input_reaction=self.reaction_anchor_map[
                                (reaction, non_fragment_indices[0])
                            ],
                            input_fragments=tuple(fragments),
                            output_molecule=state.molecule,
                        )
                        if (
                            action not in possible_actions_set
                            and self._is_action_product_selection_valid_in_forward(action)
                            and (
                                not self.use_recurrence
                                or self._is_future_composable(
                                    previous_molecule,
                                    state.num_reactions - 1,
                                    state.num_decompositions,
                                )
                            )
                        ):
                            possible_actions.append(action)
                            possible_actions_set.add(action)

        exit_action = None
        if state.budget > 0 and (
            not self.use_recurrence or self._is_composable(state.molecule, state.num_decompositions)
        ):
            exit_action = ReactionActionExit()

        if len(possible_actions) == 0 and exit_action is None:
            return ReactionActionSpace0Invalid()
        else:
            return ReactionActionSpace0orCBackward(
                possible_actions=tuple(possible_actions),
                exit_action=exit_action,
            )

    def _get_backward_action_spaces_a_standard(
        self, state: ReactionStateA
    ) -> ReactionActionSpace0orCBackward | ReactionActionSpace0Invalid:
        possible_actions: List[ReactionAction0 | ReactionActionC] = []
        if self._is_fragment(state.molecule.smiles):
            fragment_idx = self.smiles_to_fragment_idx[state.molecule.smiles]
            fragment = self.fragments[fragment_idx]
            if state.num_reactions == fragment.num_reactions:
                action_0 = self.all_actions_0[fragment_idx]
                possible_actions.append(action_0)

        possible_actions_set = set()
        if state.num_reactions > 0:
            for reaction, disconnection in zip(self.reactions, self.disconnections):
                reactants_list_list = disconnection.rdkit_rxn.RunReactants(
                    [state.molecule.rdkit_mol]
                )
                for reactants_list in reactants_list_list:
                    fragments, non_fragments, non_fragment_indices = self._lazy_is_fragment_check(
                        reactants_list
                    )
                    if len(non_fragments) > 1:
                        continue

                    if len(non_fragments) == 0:
                        for idx, previous_molecule in enumerate(fragments):
                            previous_min_num_reactions = min(1, previous_molecule.num_reactions)
                            previous_fragments = fragments[:idx] + fragments[idx + 1 :]
                            num_subtree_reactions = sum(f.num_reactions for f in previous_fragments)
                            previous_num_reactions = state.num_reactions - 1 - num_subtree_reactions
                            action = ReactionActionC(
                                input_molecule=previous_molecule,
                                input_reaction=self.reaction_anchor_map[(reaction, idx)],
                                input_fragments=tuple(previous_fragments),
                                output_molecule=state.molecule,
                            )
                            if (
                                previous_num_reactions >= previous_min_num_reactions
                                and action not in possible_actions_set
                                and self._is_action_product_selection_valid_in_forward(action)
                                and (
                                    not self.use_recurrence
                                    or self._is_decomposable(
                                        previous_molecule, previous_num_reactions
                                    )
                                )
                            ):
                                possible_actions.append(action)
                                possible_actions_set.add(action)
                    else:
                        previous_molecule = non_fragments[0]
                        if not previous_molecule.valid:
                            continue
                        num_subtree_reactions = sum(f.num_reactions for f in fragments)
                        previous_num_reactions = state.num_reactions - 1 - num_subtree_reactions
                        action = ReactionActionC(
                            input_molecule=previous_molecule,
                            input_reaction=self.reaction_anchor_map[
                                (reaction, non_fragment_indices[0])
                            ],
                            input_fragments=tuple(fragments),
                            output_molecule=state.molecule,
                        )
                        if (
                            previous_num_reactions >= 1
                            and action not in possible_actions_set
                            and self._is_action_product_selection_valid_in_forward(action)
                            and (
                                not self.use_recurrence
                                or self._is_decomposable(previous_molecule, previous_num_reactions)
                            )
                        ):
                            possible_actions.append(action)
                            possible_actions_set.add(action)

        if len(possible_actions) == 0:
            return ReactionActionSpace0Invalid()
        else:
            return ReactionActionSpace0orCBackward(
                possible_actions=tuple(possible_actions),
                exit_action=None,
            )

    def _is_fragment(self, smiles: str) -> bool:
        return smiles in self.fragment_smiles_set

    def _is_starting_molecule(self, smiles: str) -> bool:
        return smiles in self.starting_molecules_smiles_set

    def _lazy_is_fragment_check(
        self, molecules: List[Mol]
    ) -> Tuple[List[Molecule], List[Molecule], List[int]]:
        """
        Checks which of the given molecules are fragments and which need to be further decomposed.
        We don't need to check all fragments in the molecule. We can stop checking
        as soon as we more than one a non-fragment molecule, because we can only decompose one molecule.
        This way we can save few `Molecule()` calls.

        Returns:
            fragments: List of Molecule objects
            non_fragments_indices: List of indices of non-fragment molecules
        """
        fragments = []
        non_fragments = []
        non_fragments_indices = []
        for idx, mol in enumerate(molecules):
            molecule = Molecule(mol)
            if molecule.valid and self._is_fragment(molecule.smiles):
                fragment_idx = self.smiles_to_fragment_idx[molecule.smiles]
                fragments.append(self.fragments[fragment_idx])
            else:
                non_fragments.append(molecule)
                non_fragments_indices.append(idx)
                if len(non_fragments) > 1:
                    break
        return fragments, non_fragments, non_fragments_indices

    def _is_future_composable(
        self, molecule: Molecule, n_reactions: int, n_decompositions: int
    ) -> bool:
        """
        This is very similar to the `_is_decomposable` method, but the base case becomes whether
        the molecule can compose back to one of the starting molecules. After implementing the
        full `_is_composable` which is also recursive, calling `_is_future_composable` could be
        extremely expensive. (unless we have a considerable number of starting molecules)
        """
        assert self.use_recurrence
        cache_key = molecule.smiles, n_reactions, n_decompositions
        if cache_key in self.future_composable_cache:
            return self.future_composable_cache[cache_key]

        if not molecule.valid:
            self.future_composable_cache[cache_key] = False
            return False

        if n_reactions == 0:
            is_composable = self._is_composable(molecule, n_decompositions)
            self.future_composable_cache[cache_key] = is_composable
            return is_composable

        for reaction, disconnection in zip(self.reactions, self.disconnections):
            reactants_list = disconnection.rdkit_rxn.RunReactants((molecule.rdkit_mol,))
            for reactants in reactants_list:
                fragments, non_fragments, _ = self._lazy_is_fragment_check(reactants)
                if len(fragments) + len(non_fragments) == 0 or len(non_fragments) > 1:
                    continue

                if len(non_fragments) == 0:
                    for fragment in fragments:
                        if self._is_future_composable(fragment, n_reactions - 1, n_decompositions):
                            self.future_composable_cache[(molecule.smiles, n_reactions)] = True
                            return True
                else:
                    if self._is_future_composable(non_fragments[0], n_reactions - 1, n_decompositions):
                        self.future_composable_cache[(molecule.smiles, n_reactions)] = True
                        return True

        self.future_composable_cache[cache_key] = False
        return False

    def _is_composable(self, molecule: Molecule, n_decompositions: int) -> bool:
        """
        Base case: when no longer composable, we check if the molecule is one of the starting molecules.

        Induction steps: for all reactions, for all fragments, we check if the product is composable.
        """
        assert self.use_recurrence
        cache_key = molecule.smiles, n_decompositions
        if cache_key in self.composable_cache:
            return self.composable_cache[cache_key]

        if not molecule.valid:
            self.composable_cache[cache_key] = False
            return False

        if n_decompositions == 0:
            return self._is_starting_molecule(molecule.smiles)

        for anchored_reaction in self.anchored_reactions:
            anchored_pattern = anchored_reaction.anchored_pattern
            if not molecule.rdkit_mol.HasSubstructMatch(anchored_pattern.rdkit_pattern):
                continue

            fragment_patterns = anchored_reaction.fragment_patterns
            fragments_product = (
                self.pattern_to_compatible_fragments[pattern] for pattern in fragment_patterns
            )
            for fragments in itertools.product(*fragments_product):
                reactants = (molecule,) + fragments
                products = anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])
                products_list = (Molecule(mol[0]) for mol in products)
                products_list = (mol for mol in products_list if mol.valid)
                for product in products_list:
                    if self._is_composable(product, n_decompositions - 1):
                        self.composable_cache[cache_key] = True
                        return True

        self.composable_cache[cache_key] = False
        return False

    def _is_decomposable(self, molecule: Molecule, n_reactions: int) -> bool:
        """
        Recursive helper for the decompose function. Returns true if the given Molecule
        can be disconnected fully into fragments in our library.

        Base case: is the molecule a fragment in our library? If yes, we're done.

        Recursive case: For each disconnect reaction, can we use this reaction to
            disconnect our molecule? Iterate through all fragment pairs generated by
            each reaction. If any fragment pair can be decomposed, return true.

        """
        assert self.use_recurrence
        if (molecule.smiles, n_reactions) in self.recurrence_cache:
            return self.recurrence_cache[(molecule.smiles, n_reactions)]

        if not molecule.valid:
            self.recurrence_cache[(molecule.smiles, n_reactions)] = False
            return False

        if n_reactions == 0:
            return self._is_fragment(molecule.smiles)

        # Decompose the molecule by all reverse reactions possible.
        # If ANY of them work then return True.
        for reaction, disconnection in zip(self.reactions, self.disconnections):
            reactants_list = disconnection.rdkit_rxn.RunReactants((molecule.rdkit_mol,))
            for reactants in reactants_list:
                fragments, non_fragments, _ = self._lazy_is_fragment_check(reactants)
                if len(fragments) + len(non_fragments) == 0 or len(non_fragments) > 1:
                    continue

                if len(non_fragments) == 0:
                    for fragment in fragments:
                        if self._is_decomposable(fragment, n_reactions - 1):
                            self.recurrence_cache[(molecule.smiles, n_reactions)] = True
                            return True
                    continue

                if self._is_decomposable(non_fragments[0], n_reactions - 1):
                    output_products = reaction.rdkit_rxn.RunReactants(
                        [r.rdkit_mol for r in reactants]
                    )
                    for mol in output_products:
                        if molecule == Molecule(mol[0]):
                            self.recurrence_cache[(molecule.smiles, n_reactions)] = True
                            return True

        # We were unable to find a fully decomposable fragment pair.
        self.recurrence_cache[(molecule.smiles, n_reactions)] = False
        return False

    def _is_decomposable(self, molecule: Molecule, n_reactions: int) -> bool:
        """
        Recursive helper for the decompose function. Returns true if the given Molecule
        can be disconnected fully into fragments in our library.

        Base case: is the molecule a fragment in our library? If yes, we're done.

        Recursive case: For each disconnect reaction, can we use this reaction to
            disconnect our molecule? Iterate through all fragment pairs generated by
            each reaction. If any fragment pair can be decomposed, return true.

        """
        assert self.use_recurrence
        if (molecule.smiles, n_reactions) in self.recurrence_cache:
            return self.recurrence_cache[(molecule.smiles, n_reactions)]

        if (
            self._is_fragment(molecule.smiles)
            and self.fragments[self.smiles_to_fragment_idx[molecule.smiles]].num_reactions
            == n_reactions
        ):
            return True

        if n_reactions == 0:
            return False

        if not molecule.valid:
            self.recurrence_cache[(molecule.smiles, n_reactions)] = False
            return False

        # Decompose the molecule by all reverse reactions possible.
        # If ANY of them work then return True.
        for reaction, disconnection in zip(self.reactions, self.disconnections):
            reactants_list = disconnection.rdkit_rxn.RunReactants((molecule.rdkit_mol,))
            for reactants in reactants_list:
                fragments, non_fragments, _ = self._lazy_is_fragment_check(reactants)
                if len(fragments) + len(non_fragments) == 0 or len(non_fragments) > 1:
                    continue

                sub_tree_reactions = sum(f.num_reactions for f in fragments)
                if len(non_fragments) == 0:
                    for fragment in fragments:
                        # we want to ensure that there is at least one fragment (coming from the library) in `reactants`
                        # that can be further decomposed with n_reactions - 1 reactions.
                        subtree_remaining = sub_tree_reactions - fragment.num_reactions
                        if self._is_decomposable(fragment, n_reactions - 1 - subtree_remaining):
                            self.recurrence_cache[(molecule.smiles, n_reactions)] = True
                            return True
                    continue

                if self._is_decomposable(non_fragments[0], n_reactions - sub_tree_reactions - 1):
                    # We should check the compatibility of forward and backward reaction here
                    # However, it's computationally expensive and we make another hack to
                    # get rid of empty action space issue in the backward sampling process.

                    # output_products = reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])
                    # for mol in output_products:
                    #     if molecule == Molecule(mol[0]):
                    #         self._cache[(molecule.smiles, n_reactions)] = True
                    #         return True

                    self.recurrence_cache[(molecule.smiles, n_reactions)] = True
                    return True

        # We were unable to find a fully decomposable fragment pair.
        self.recurrence_cache[(molecule.smiles, n_reactions)] = False
        return False

    def _get_backward_action_spaces_b(
        self, state: ReactionStateB
    ) -> ReactionActionSpaceA | ReactionActionSpaceB:
        if self.starting_molecules_smiles_set:
            return self._get_backward_action_spaces_b_optimization(state)
        else:
            return self._get_backward_action_spaces_b_standard(state)

    def _get_backward_action_spaces_b_optimization(
        self, state: ReactionStateB
    ) -> ReactionActionSpaceA | ReactionActionSpaceB:
        exit_action = None

        if state.num_decompositions > 0:
            unfilled_patterns = state.anchored_reaction.fragment_patterns[len(state.fragments):]
            if len(unfilled_patterns) > 0:
                unfilled_fragments_product = (
                    self.pattern_to_compatible_fragments[pattern] for pattern in unfilled_patterns
                )
                for unfilled_fragments in itertools.product(*unfilled_fragments_product):
                    reactants = (state.molecule,) + state.fragments + unfilled_fragments
                    products = state.anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])
                    products_list = (Molecule(mol[0]) for mol in products)
                    products_list = (mol for mol in products_list if mol.valid)
                    for product in products_list:
                        if not self.use_recurrence or self._is_composable(product, state.num_decompositions - 1):
                            exit_action = ReactionActionExit()
                            break
                    if exit_action is not None:
                        break
            else:
                reactants = (state.molecule,) + state.fragments
                products = state.anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])
                products_list = (Molecule(mol[0]) for mol in products)
                products_list = (mol for mol in products_list if mol.valid)
                for product in products_list:
                    if not self.use_recurrence or self._is_composable(product, state.num_decompositions - 1):
                        exit_action = ReactionActionExit()
                        break
        else:
            exit_action = ReactionActionExit()

        if len(state.fragments) == 0:
            mask = [False] * len(self.all_actions_a)
            mask[state.anchored_reaction.idx] = True
            return ReactionActionSpaceA(
                all_actions=self.all_actions_a,
                possible_actions_mask=mask,
                exit_action=exit_action,
            )
        previous_fragment_smiles = state.fragments[-1].smiles
        previous_fragment = self.fragments[self.smiles_to_fragment_idx[previous_fragment_smiles]]
        return ReactionActionSpaceB(
            possible_actions=(self.all_actions_b[previous_fragment.idx],),
            exit_action=exit_action,
        )

    def _get_backward_action_spaces_b_standard(
        self, state: ReactionStateB
    ) -> ReactionActionSpaceA | ReactionActionSpaceB:
        if len(state.fragments) == 0:
            mask = [False] * len(self.all_actions_a)
            mask[state.anchored_reaction.idx] = True
            return ReactionActionSpaceA(
                all_actions=self.all_actions_a,
                possible_actions_mask=mask,
                exit_action=None,
            )
        previous_fragment_smiles = state.fragments[-1].smiles
        previous_fragment = self.fragments[self.smiles_to_fragment_idx[previous_fragment_smiles]]
        return ReactionActionSpaceB(
            possible_actions=(self.all_actions_b[previous_fragment.idx],),
            exit_action=None,
        )

    def _get_backward_action_spaces_c(
        self, state: ReactionStateC
    ) -> ReactionActionSpaceA | ReactionActionSpaceB:
        state_b = ReactionStateB(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            fragments=state.fragments,
            num_reactions=state.num_decompositions,
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )
        return self._get_backward_action_spaces_b(state_b)

    def _get_backward_action_spaces_terminal(
        self, state: ReactionStateTerminal
    ) -> ReactionActionSpaceA:
        mask = [False] * len(self.all_actions_a)
        mask[-1] = True
        return ReactionActionSpaceA(
            all_actions=self.all_actions_a,
            possible_actions_mask=mask,
            exit_action=None,
        )

    def _get_backward_action_spaces_early_terminal(
        self, state: ReactionStateEarlyTerminal
    ) -> ReactionActionSpace:
        return ReactionActionSpaceEarlyTerminate()

    def apply_forward_actions(
        self, states: List[ReactionState], actions: List[ReactionAction]
    ) -> List[ReactionState]:
        new_states = []
        for state, action in zip(states, actions):
            action_type = type(action)
            new_state = self.forward_action_dict[action_type](state, action)
            new_states.append(new_state)
        return new_states

    def _apply_forward_actions_h(
        self, state: ReactionStateH, action: ReactionActionH
    ) -> ReactionStateX:
        return ReactionStateX(
            molecule=action.fragment,
            num_decompositions=0,
            budget=action.budget
        )

    def _apply_forward_actions_x(
        self, state: ReactionStateX, action: ReactionActionX
    ) -> ReactionStateY:
        return ReactionStateY(
            molecule=action.input_molecule,
            anchored_reaction=action.input_reaction,
            fragments=action.input_fragments,
            num_decompositions=state.num_decompositions + 1,
            budget=state.budget + 1,
        )

    def _apply_forward_actions_y(
        self, state: ReactionStateY | ReactionStateZ, action: ReactionActionY
    ) -> ReactionStateZ:
        last_fragment = self.fragments[self.smiles_to_fragment_idx[state.fragments[-1].smiles]]
        assert last_fragment.num_reactions == state.fragments[-1].num_reactions
        return ReactionStateZ(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            fragments=state.fragments[:-1],
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_forward_actions_z(
        self, state: ReactionStateY | ReactionStateZ, action: ReactionActionZ
    ) -> ReactionStateX:
        return ReactionStateX(
            molecule=state.molecule,
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_forward_actions_exit(
        self, state: ReactionStateX | ReactionStateY | ReactionStateZ, action: ReactionActionExit
    ) -> ReactionStateA | ReactionStateC | ReactionStateB:
        if isinstance(state, ReactionStateX):
            return ReactionStateA(
                molecule=state.molecule,
                num_reactions=state.molecule.num_reactions,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        elif isinstance(state, ReactionStateY):
            return ReactionStateC(
                molecule=state.molecule,
                anchored_reaction=state.anchored_reaction,
                fragments=state.fragments,
                num_reactions=state.molecule.num_reactions,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        elif isinstance(state, ReactionStateZ):
            return ReactionStateB(
                molecule=state.molecule,
                anchored_reaction=state.anchored_reaction,
                fragments=state.fragments,
                num_reactions=state.molecule.num_reactions,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        else:
            assert False

    def _apply_forward_actions_0(
        self, state: ReactionState0, action: ReactionAction0
    ) -> ReactionStateA:
        return ReactionStateA(
            molecule=action.fragment,
            num_reactions=action.fragment.num_reactions,
            num_decompositions=0,
            budget=0
        )

    def _apply_forward_actions_0_invalid(
        self, state: ReactionState0Invalid, action: ReactionAction0Invalid
    ) -> ReactionState0Invalid:
        return state.previous_state

    def _apply_forward_actions_a(
        self, state: ReactionStateA, action: ReactionActionA
    ) -> ReactionStateB | ReactionStateC | ReactionStateTerminal:
        if action.anchored_reaction is None:
            return ReactionStateTerminal(
                molecule=state.molecule,
                num_reactions=state.num_reactions,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        if len(action.anchored_reaction.fragment_patterns) == 0:
            return ReactionStateC(
                molecule=state.molecule,
                anchored_reaction=action.anchored_reaction,
                fragments=(),
                num_reactions=state.num_reactions,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        return ReactionStateB(
            molecule=state.molecule,
            anchored_reaction=action.anchored_reaction,
            num_reactions=state.num_reactions,
            fragments=(),
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_forward_actions_b(
        self, state: ReactionStateB, action: ReactionActionB
    ) -> ReactionStateB | ReactionStateC:
        new_fragments = state.fragments + (action.fragment,)
        num_reactions = state.num_reactions + action.fragment.num_reactions
        if len(new_fragments) == len(state.anchored_reaction.fragment_patterns):
            return ReactionStateC(
                molecule=state.molecule,
                anchored_reaction=state.anchored_reaction,
                fragments=new_fragments,
                num_reactions=num_reactions,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        return ReactionStateB(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            fragments=new_fragments,
            num_reactions=num_reactions,
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_forward_actions_c(
        self, state: ReactionStateC, action: ReactionActionC
    ) -> ReactionStateA:
        return ReactionStateA(
            molecule=action.output_molecule,
            num_reactions=state.num_reactions + 1,
            num_decompositions=state.num_decompositions,
            budget=state.budget - 1,
        )

    def _apply_forward_actions_early_terminate(
        self, state: ReactionState, action: ReactionActionEarlyTerminate
    ) -> ReactionStateEarlyTerminal:
        return ReactionStateEarlyTerminal(previous_state=state)

    def apply_backward_actions(
        self, states: List[ReactionState], actions: List[ReactionAction]
    ) -> List[ReactionState]:
        new_states = []
        for state, action in zip(states, actions):
            action_type = type(action)
            new_state = self.backward_action_dict[action_type](state, action)
            new_states.append(new_state)
        return new_states

    def _apply_backward_actions_h(
        self, state: ReactionStateX, action: ReactionActionH
    ) -> ReactionStateH:
        return ReactionStateH()

    def _apply_backward_actions_x(
        self, state: ReactionStateY, action: ReactionActionX
    ) -> ReactionStateX:
        return ReactionStateX(
            molecule=action.output_molecule,
            num_decompositions=state.num_decompositions - 1,
            budget=state.budget - 1,
        )

    def _apply_backward_actions_y(
        self, state: ReactionStateY | ReactionStateZ, action: ReactionActionY
    ) -> ReactionStateY | ReactionStateZ:
        new_fragments = state.fragments + (action.fragment,)
        if len(new_fragments) == len(state.anchored_reaction.fragment_patterns):
            return ReactionStateY(
                molecule=state.molecule,
                anchored_reaction=state.anchored_reaction,
                fragments=new_fragments,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        return ReactionStateZ(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            fragments=new_fragments,
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_backward_actions_z(
        self, state: ReactionStateX, action: ReactionActionZ
    ) -> ReactionStateY | ReactionStateZ:
        if len(action.anchored_reaction.fragment_patterns) == 0:
            return ReactionStateY(
                molecule=state.molecule,
                anchored_reaction=action.anchored_reaction,
                fragments=(),
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        return ReactionStateZ(
            molecule=state.molecule,
            anchored_reaction=action.anchored_reaction,
            fragments=(),
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_backward_actions_exit(
        self, state: ReactionStateA | ReactionStateB | ReactionStateC, action: ReactionActionExit
    ) -> ReactionStateX | ReactionStateZ | ReactionStateY:
        if isinstance(state, ReactionStateA):
            return ReactionStateX(
                molecule=state.molecule,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        elif isinstance(state, ReactionStateB):
            return ReactionStateZ(
                molecule=state.molecule,
                anchored_reaction=state.anchored_reaction,
                fragments=state.fragments,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        elif isinstance(state, ReactionStateC):
            return ReactionStateY(
                molecule=state.molecule,
                anchored_reaction=state.anchored_reaction,
                fragments=state.fragments,
                num_decompositions=state.num_decompositions,
                budget=state.budget,
            )
        else:
            assert False

    def _apply_backward_actions_0(
        self, state: ReactionStateA, action: ReactionAction0
    ) -> ReactionState0:
        return ReactionState0()

    def _apply_backward_actions_0_invalid(
        self, state: ReactionState, action: ReactionAction0Invalid
    ) -> ReactionState0Invalid:
        return ReactionState0Invalid(previous_state=state)

    def _apply_backward_actions_a(
        self,
        state: ReactionStateB | ReactionStateC | ReactionStateTerminal,
        action: ReactionActionA,
    ) -> ReactionStateA:
        return ReactionStateA(
            molecule=state.molecule,
            num_reactions=state.num_reactions,
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_backward_actions_b(
        self, state: ReactionStateB | ReactionStateC, action: ReactionActionB
    ) -> ReactionStateB:
        last_fragment = self.fragments[self.smiles_to_fragment_idx[state.fragments[-1].smiles]]
        assert last_fragment.num_reactions == state.fragments[-1].num_reactions
        return ReactionStateB(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            num_reactions=state.num_reactions - last_fragment.num_reactions,
            fragments=state.fragments[:-1],
            num_decompositions=state.num_decompositions,
            budget=state.budget,
        )

    def _apply_backward_actions_c(
        self, state: ReactionStateA, action: ReactionActionC
    ) -> ReactionStateC:
        return ReactionStateC(
            molecule=action.input_molecule,
            anchored_reaction=action.input_reaction,
            fragments=action.input_fragments,
            num_reactions=state.num_reactions - 1,
            num_decompositions=state.num_decompositions,
            budget=state.budget + 1,
        )

    def _apply_backward_actions_early_terminate(
        self, state: ReactionStateEarlyTerminal, action: ReactionActionEarlyTerminate
    ) -> ReactionState:
        return state.previous_state

    def get_terminal_mask(self, states: List[ReactionState]) -> List[bool]:
        return [
            isinstance(state, (ReactionStateTerminal, ReactionStateEarlyTerminal))
            for state in states
        ]

    def get_source_mask(self, states: List[ReactionState]) -> List[bool]:
        return [isinstance(state, (ReactionState0, ReactionStateH, ReactionState0Invalid)) for state in states]

    def get_valid_source_mask(self, states: List[TState]) -> List[bool]:
        return [isinstance(state, (ReactionState0, ReactionStateH)) for state in states]

    def sample_source_states(self, n_states: int) -> List[ReactionState]:
        if self.starting_molecules_smiles_set:
            return [ReactionStateH()] * n_states
        else:
            return [ReactionState0()] * n_states

    def sample_terminal_states(self, n_states: int) -> List[ReactionState]:
        raise NotImplementedError()

    def get_num_source_states(self) -> int:
        raise NotImplementedError()

    def get_source_states_at_index(self, index: List[int]) -> List[TState]:
        raise NotImplementedError()

    def get_num_terminal_states(self) -> int:
        raise NotImplementedError()

    def get_terminal_states_at_index(self, index: List[int]) -> List[TState]:
        raise NotImplementedError()
