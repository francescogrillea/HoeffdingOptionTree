from collections import Counter, deque
import logging

from river.tree import HoeffdingTreeClassifier
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch
from option_node import OptionNode

logging.basicConfig(
    level=logging.INFO,
    format='[in %(funcName)s] %(message)s'
)

logger = logging.getLogger(__name__)


class HoeffdingOptionTreeClassifier(HoeffdingTreeClassifier):
    """ Hoeffding Option Tree classifier.
    """

    def __init__(self, delta_prime: float = 0.955, max_options=3, **kwargs):
        super().__init__(**kwargs)
        self._delta_prime = delta_prime
        self.max_options = max_options
        self.count = 0

        self._root: DTBranch | HTLeaf | OptionNode = None

    def learn_one(self, x, y, *, w=1.0):

        self.count += 1
        logger.info(f"{self.count}\tNew instance received. Class: {y}")
        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            current_node = self._new_leaf()
            self._root = OptionNode(self.max_options,
                                    initial_node=current_node)

        nodes_to_traverse = deque()
        for child in self._root.children:
            nodes_to_traverse.append((child, self._root))

        if self._root.candidate_option_branch is not None:
            nodes_to_traverse.append((self._root.candidate_option_branch, self._root))

        # traverse until leaf(s)
        while nodes_to_traverse:
            current_node, parent_node = nodes_to_traverse.popleft()

            if isinstance(parent_node, OptionNode) and parent_node.can_add_candidate():
                new_candidate = self._new_leaf()
                parent_node.candidate_option_branch = new_candidate
                nodes_to_traverse.append((parent_node.candidate_option_branch, parent_node))

            while isinstance(current_node, DTBranch):
                p_branch = parent_node.branch_no(x) if isinstance(parent_node, DTBranch) else None
                parent_node = current_node
                current_node = current_node.next(x)

            if isinstance(current_node, OptionNode):
                for child in current_node.children:
                    nodes_to_traverse.append((child, current_node))
                if current_node.candidate_option_branch is not None:
                    nodes_to_traverse.append((current_node.candidate_option_branch, current_node))

            if isinstance(current_node, HTLeaf):
                current_node.learn_one(x, y, w=w, tree=self)
                if self._growth_allowed:
                    weight_seen = current_node.total_weight
                    weight_diff = weight_seen - current_node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = parent_node.branch_no(x) if isinstance(parent_node, DTBranch) else None
                        self._attempt_to_split(current_node, parent_node, p_branch)
                        current_node.last_split_attempt_at = weight_seen

    def traverse(self, x):

        active_nodes = []
        active_nodes += self._root.children
        if self._root.candidate_option_branch is not None:
            active_nodes.append(self._root.candidate_option_branch)

        leafs = []
        while len(active_nodes) > 0:
            starting_node = active_nodes.pop(0)
            if isinstance(starting_node, OptionNode):
                logger.info("Serve?")
                active_nodes += starting_node.children

            elif isinstance(starting_node, DTBranch):
                node = starting_node.next(x)
                while isinstance(node, DTBranch):
                    node = node.next(x)

                if isinstance(node, OptionNode):
                    active_nodes += node.children
                    if node.candidate_option_branch is not None:
                        active_nodes.append(node.candidate_option_branch)
                elif isinstance(node, HTLeaf):
                    leafs.append(node)

            elif isinstance(starting_node, HTLeaf):
                leafs.append(starting_node)
            else:
                raise TypeError(f"Type {type(starting_node)} not supported")
        return leafs

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in sorted(self.classes)}
        if self._root is not None:
            leafs = self.traverse(x)
            leaf = Counter(leafs).most_common(1)[0][0]
            proba.update(leaf.prediction(x, tree=self))
        return proba

    def _attempt_to_split(self, leaf: HTLeaf, parent: DTBranch | OptionNode, parent_branch: int, **kwargs):

        if leaf.observed_class_distribution_is_pure():  # type: ignore
            return

        split_criterion = self._new_split_criterion()
        best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
        best_split_suggestions.sort()
        should_split = False

        if parent.has_children():
            if parent.option_count() == self.max_options:
                return

            best_split_suggestions = [suggestion for suggestion in best_split_suggestions if suggestion.feature not in parent.split_features and suggestion.feature is not None]

            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self._delta_prime,
                    leaf.total_weight,
                )

                best_attribute_merit = parent.bestG
                split_decision = best_split_suggestions[-1]

                if split_decision.merit - best_attribute_merit > hoeffding_bound:
                    should_split = True

            if should_split:
                split_decision = best_split_suggestions[-1]
                # logger.info(f"Adding new DTBranch to OptionNode, which splits on {split_decision.feature}")

                branch = self._branch_selector(
                    split_decision.numerical_feature, split_decision.multiway_split
                )
                leaves = tuple(
                    self._new_leaf(initial_stats, parent=leaf)
                    for initial_stats in split_decision.children_stats  # type: ignore
                )

                new_split = split_decision.assemble(
                    branch, leaf.stats, leaf.depth, *leaves, **kwargs)

                for i, child in enumerate(new_split.children):
                    new_split.children[i] = OptionNode(max_option=self.max_options,
                                                       initial_node=new_split.children[i])
                parent.add_option(new_split, split_decision)
        else:

            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )

                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                        best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tau
                ):
                    should_split = True

            if should_split:
                split_decision = best_split_suggestions[-1]
                logger.info(f"Split on {split_decision.feature}")

                if split_decision.feature is not None:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs)

                    for i, child in enumerate(new_split.children):
                        new_split.children[i] = OptionNode(max_option=self.max_options,
                                                           initial_node=new_split.children[i])

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)

                    parent.add_option(new_split, split_decision)
