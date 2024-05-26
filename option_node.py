from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch


class OptionNode:
    _traverse_mode = ["first", "all"]

    def __init__(self, max_option,
                 initial_node: HTLeaf = None):

        self.max_option = max_option
        self.children: list[DTBranch] = []
        self._candidate_option_branch = initial_node

        self.split_features = set()
        self.split_suggestions = None

        self._bestG = 0
        self._bestG_index = -1
        self._hoeffding_bound = 0

    def add_option(self, x: DTBranch, split_decision):
        self.children.append(x)

        self.split_features.add(split_decision.feature)
        self.candidate_option_branch = None

        if self._bestG < split_decision.merit:
            self._bestG = split_decision.merit

        # clean statistics
        if len(self.children) == self.max_option:
            delattr(self, "split_suggestions")
            delattr(self, "_hoeffding_bound")

    def attempt_cleanup_stats(self):
        if len(self.children) == self.max_option:
            delattr(self, "split_suggestions")
            delattr(self, "_hoeffding_bound")

    def update_option_stats(self, hoeffding_bound: float, split_attribute: str, split_suggestions: list):
        self.split_features.add(split_attribute)
        self._hoeffding_bound = hoeffding_bound
        self.split_suggestions = split_suggestions
        # update _bestG and _bestG_index
        i, new_g = [(i, suggestion.merit) for i, suggestion in enumerate(split_suggestions) if
                    suggestion.feature is not None and suggestion.feature == split_attribute][0]
        if self._bestG < new_g:
            self._bestG = new_g
            self._bestG_index = i

    def has_only_leaf(self):
        return len(self.children) == 1 and isinstance(self.children[0], HTLeaf)

    def has_children(self):
        return len(self.children) > 0 and not isinstance(self.children[0], HTLeaf)

    def option_count(self):
        return len(self.children)

    def traverse(self, mode="first"):
        # TODO - controllare se va bene passarla per riferimento
        return self.children

    def can_split(self):
        return len(self.children) < self.max_option

    @property
    def hoeffding_bound(self):
        return self._hoeffding_bound

    @hoeffding_bound.setter
    def hoeffding_bound(self, value):
        self._hoeffding_bound = value

    @property
    def bestG(self):
        return self._bestG

    @property
    def candidate_option_branch(self):
        return self._candidate_option_branch

    @candidate_option_branch.setter
    def candidate_option_branch(self, value):
        self._candidate_option_branch = value

    def is_leaf_candidate(self, leaf: HTLeaf):
        return self.candidate_option_branch == leaf

    def can_add_candidate(self):
        return self.candidate_option_branch is None and self.max_option > len(self.children) > 0
