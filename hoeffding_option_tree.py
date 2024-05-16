from collections import Counter

from river.tree.splitter import GaussianSplitter
from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from river.tree import HoeffdingTreeClassifier
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch
from option_node import OptionNode

class HoeffdingOptionTreeClassifier(HoeffdingTreeClassifier):
    """ Hoeffding Option Tree classifier.
    """

    def __init__(self, delta_prime=0.0, max_options=3, **kwargs):
        super().__init__(**kwargs)
        self.delta_prime = delta_prime
        self.max_options = max_options

        self._root: DTBranch | HTLeaf | OptionNode = None
        self._subtree_root: DTBranch | HTLeaf = None

    def learn_one(self, x, y, *, w=1.0):
        # Updates the set of observed classes
        print(f"Model received input x and label y={y}")
        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = OptionNode()
            self._n_active_leaves = 0

        print(f"Root Class: {self._root.__class__.__name__}")

        parent_node = None
        current_node = None

        for child in self._root.children:
            current_node = child
            parent_node = None

            if isinstance(current_node, DTBranch):
                print(f"Current node is a DTBranch")
                path = iter(child.walk(x, until_leaf=False))
                print(path)
                while True:
                    aux = next(path, None)
                    if aux is None:
                        break
                    parent_node = current_node
                    current_node = aux

            self._subtree_root = current_node

            if isinstance(current_node, HTLeaf):
                print(f"Current node is a HTLeaf")
                current_node.learn_one(x, y, w=w, tree=self)
                p_branch = parent_node.branch_no(x) if isinstance(parent_node, DTBranch) else None
                self._attempt_to_split(current_node, parent_node, p_branch)
                print(f"Current node after learn_one: {current_node.stats}")
                # TODO - ci deve essere un attempts to option?

            elif isinstance(current_node, OptionNode):
                print(f"Current node is a OptionNode")
                # TODO - che devo fare?
            elif isinstance(current_node, DTBranch):
                print(f"Current node is a DTBranch")
            else:
                raise TypeError

        # if no option branches are present
        if current_node is None:
            new_node = self._new_leaf()
            new_node.learn_one(x, y, w=w, tree=self)
            print(f"No branches are present. New leaf created\t {new_node.stats}.")
            self._root.add_option_branch(new_node)
            self._n_active_leaves = 1

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in sorted(self.classes)}
        active_nodes = [self._root]
        leafs = []
        while len(active_nodes) > 0:
            starting_node = active_nodes.pop(0)
            if isinstance(starting_node, OptionNode):
                active_nodes.append(starting_node.traverse())
            # TODO - non mi torna come fa a richiamare la traverse su un option node visto che arriva alla foglia!
            elif isinstance(starting_node, DTBranch):
                leaf = starting_node.traverse(x, until_leaf=True)
                leafs.append(leaf)
            else:
                leaf = starting_node
                leafs.append(leaf)

        leaf = Counter(leafs).most_common(1)[0][0]
        proba.update(leaf.prediction(x, tree=self))
        return proba

    # TODO - gestire il fatto di creare un option node
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
                print("SHOULD SPLIT")
                split_decision = best_split_suggestions[-1]
                if split_decision.feature is None:
                    # Pre-pruning - null wins
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._subtree_root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                # Manage memory
                # self._enforce_size_limit()