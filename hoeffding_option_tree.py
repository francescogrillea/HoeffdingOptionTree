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

    def __init__(self, delta_prime=0.0, split_option_criterion="random", max_options=3, **kwargs):
        super().__init__(**kwargs)
        self.delta_prime = delta_prime
        self.max_options = max_options
        self.count = 0
        self.buffer = []

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
        if isinstance(self._root, OptionNode):  # sempre true ?
            for child in self._root.children:
                nodes_to_traverse.append((child, self._root))
        else:
            nodes_to_traverse.append((self._root, None))

        # traverse until leaf(s)
        while nodes_to_traverse:
            current_node, parent_node = nodes_to_traverse.popleft()
            while isinstance(current_node, DTBranch):
                p_branch = parent_node.branch_no(x) if isinstance(parent_node, DTBranch) else None
                parent_node = current_node
                current_node = current_node.next(x)

            if isinstance(current_node, OptionNode):
                for child in current_node.children:
                    nodes_to_traverse.append((child, current_node))
                # nodes_to_traverse.extend(current_node.children)
                self._attempt_to_split_option(x, y, w, current_node, parent_node)

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
            leafs = self.traverse(x)
            leaf = Counter(leafs).most_common(1)[0][0]
            proba.update(leaf.prediction(x, tree=self))
        return proba

    def _attempt_to_split(self, leaf: HTLeaf, parent: DTBranch | OptionNode, parent_branch: int, **kwargs):

        if not leaf.observed_class_distribution_is_pure():  # type: ignore

            split_criterion = self._new_split_criterion()
            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False

            # CASO 1 -> 2
            if isinstance(parent, OptionNode) and parent.has_only_leaf():

                parent_branch = 0
                hoeffding_bound = -1

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
                logger.info(f"ShouldSplit: {should_split}")
                if should_split:
                    split_decision = best_split_suggestions[-1]
                    logger.info(
                        f"Split on {split_decision.feature}\t Parent: {parent.feature if isinstance(parent, DTBranch) else parent.__class__.__name__}")
                    # logger.info(f"LeafNode: {leaf.stats}")
                    # logger.info(f"ParentNode: {parent.feature if isinstance(parent, DTBranch) else parent.__class__.__name__}")
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

                        new_split = split_decision.assemble(
                            branch, leaf.stats, leaf.depth, *leaves, **kwargs)
                        accumulated_features = parent.features_along_path
                        accumulated_features.add(split_decision.feature)
                        for i, child in enumerate(new_split.children):
                            new_split.children[i] = OptionNode(max_option=self.max_options,
                                                               initial_node=new_split.children[i])
                            # new_split.children[i].update_option_stats(hoeffding_bound=hoeffding_bound,
                            #                                           split_attribute=new_split.feature,
                            #                                           split_suggestions=best_split_suggestions)
                        self._n_active_leaves -= 1
                        self._n_active_leaves += len(leaves)
                        if parent is None:  # non entra mai ?
                            self._root = new_split
                        else:
                            # inject statistics into OptionNode
                            parent.update_option_stats(hoeffding_bound=hoeffding_bound,
                                                       split_attribute=new_split.feature,
                                                       split_suggestions=best_split_suggestions)

                            parent.children[parent_branch] = new_split

            elif isinstance(parent, OptionNode) and parent.has_children():
                # TODO
                pass

    def _attempt_to_split_option(self, x, y, w, current_node: OptionNode, parent_node: DTBranch | None, **kwargs):

        if len(current_node.children) < current_node.max_option and not current_node.has_only_leaf():
            feasible_suggestions = [suggestion for suggestion in current_node.split_suggestions if
                                    suggestion.feature not in list(current_node.features_along_path)]
            split_decision = max(feasible_suggestions, key=lambda suggestion: suggestion.merit)
            g_x = split_decision.merit
            # logger.info(f"GX: {g_x}\t CurrentBestG: {current_node.bestG}\t HB: {current_node.hoeffding_bound}")
            if g_x - current_node.bestG > current_node.hoeffding_bound:
                branch = self._branch_selector(
                    split_decision.numerical_feature, split_decision.multiway_split
                )
                new_leaf = self._new_leaf()
                new_leaf.learn_one(x, y, w=w, tree=self)
                new_split = split_decision.assemble(
                    branch, new_leaf.stats, new_leaf.depth, **kwargs)
                #                branch, new_leaf.stats, new_leaf.depth, *leaves, ** kwargs)

                current_node.add_option(new_split)
