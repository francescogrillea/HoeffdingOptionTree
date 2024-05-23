import random
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

    def __init__(self, delta_prime=0.0, max_options=3, **kwargs):
        super().__init__(**kwargs)
        self.delta_prime = delta_prime
        self.max_options = max_options
        self.count = 0

        self._root: DTBranch | HTLeaf | OptionNode = None
        self._subtree_root: DTBranch | HTLeaf = None

    def learn_one(self, x, y, *, w=1.0):

        self.count += 1
        logger.info(f"{self.count}\tNew instance received. Class: {y}")
        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            new_node = self._new_leaf()
            new_node.learn_one(x, y, w=w, tree=self)
            self._root = new_node
            self._n_active_leaves = 1
            # TO eliminate
            return

        current_node = self._root
        parent_node = None
        #

        nodes_to_traverse = deque()
        if isinstance(current_node, OptionNode):
            parent_node = current_node
            nodes_to_traverse.extend(current_node.children)
        else:
            nodes_to_traverse.append(current_node)

        while nodes_to_traverse:
            current_node = nodes_to_traverse.popleft()
            while isinstance(current_node, DTBranch):
                p_branch = parent_node.branch_no(x) if isinstance(parent_node, DTBranch) else None

                self._attempt_to_split_option(x, y, w, current_node, parent_node, p_branch)
                parent_node = current_node
                current_node = current_node.next(x)

            if isinstance(current_node, OptionNode):
                nodes_to_traverse.extend(current_node.children)

            if isinstance(current_node, HTLeaf):
                current_node.learn_one(x, y, w=w, tree=self)
                if self._growth_allowed:
                    p_branch = parent_node.branch_no(x) if isinstance(parent_node, DTBranch) else None
                    self._attempt_to_split(current_node, parent_node, p_branch)

    def traverse(self, x):

        active_nodes = []

        if isinstance(self._root, OptionNode):
            active_nodes += self._root.children
        else:
            active_nodes += [self._root]

        leafs = []
        while len(active_nodes) > 0:
            starting_node = active_nodes.pop(0)
            if isinstance(starting_node, OptionNode):
                active_nodes += starting_node.children
            elif isinstance(starting_node, DTBranch):
                node = starting_node.next(x)
                while isinstance(node, DTBranch):
                    node = node.next(x)

                if isinstance(node, OptionNode):
                    active_nodes += node.children
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
            leafs = self.traverse(x)  # TODO - errore qui
            leaf = Counter(leafs).most_common(1)[0][0]
            proba.update(leaf.prediction(x, tree=self))
        return proba

    def _attempt_to_split(self, leaf: HTLeaf, parent: DTBranch, parent_branch: int, **kwargs):
        if not leaf.observed_class_distribution_is_pure():  # type: ignore

            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False
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
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        if (
                                suggestion.feature
                                and best_suggestion.merit - suggestion.merit > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                logger.info(f"Should Split on attribute {split_decision.feature}")
                # if split_decision.feature is None:
                #     # Pre-pruning - null wins
                #     leaf.deactivate()
                #     self._n_inactive_leaves += 1
                #     self._n_active_leaves -= 1
                # else:
                if split_decision.feature is not None:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )
                    # inject the split_info to each branch
                    accumulated_features = set()
                    accumulated_features.add(split_decision.feature)
                    if parent is not None and hasattr(parent, "accumulated_features"):
                        accumulated_features.update(parent.accumulated_features)

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs,
                        split_info=split_decision.split_info,
                        accumulated_features=accumulated_features)  # TODO - check if is split_info or merit

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

    def _attempt_to_split_option(self, x, y, w, current_node: DTBranch, parent: DTBranch | OptionNode,
                                 parent_branch: int, **kwargs):
        # if leaf class distribution is not pure
        # pass the current_node variable by assignment, in order to transform it into a OptionNode
        should_split = True
        if isinstance(parent, OptionNode):
            should_split = parent.option_count() < self.max_options

        if should_split:
            random.seed(123)
            # if option node should be added
            if random.random() < 0.07:  # TODO - cambiare
                logger.info(f"Should Split on OptionNode on {current_node.feature}")

                # TODO - non va aggiunta una foglia ma un DTBranch
                new_node = self._new_leaf()
                new_node.learn_one(x, y, w=w, tree=self)

                # append the new branch to the option node above
                if isinstance(parent, OptionNode):
                    parent.add_option_branch(new_node)
                # create a new option node with two branches
                else:
                    option_node = OptionNode(initial_node=current_node)
                    option_node.add_option_branch(new_node)

                    if parent is None:
                        self._root = option_node
                    else:
                        parent.children[parent_branch] = option_node
