from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch


class OptionNode:
    """
    Option Node for Hoeffding Option Tree.

    This class represents an option node in the Hoeffding Option Tree, which can hold multiple decision branches
    in order to test simultaneously on each split node.

    Attributes:
        max_option (int): Maximum number of decision branches can be added to this node.
        initial_node (DTBranch | HTLeaf): The initial node to start with.
        children (list): List of all decision branches that can be traversed simultaneously.
        candidate_option_branch (DTBranch | HTLeaf): Candidate branch which can be added as child.
        split_features (set): Set of features used for splitting in the child branches.
        _bestG (float): Best split value among all decision branches.
    """

    def __init__(self, max_option,
                 initial_node: HTLeaf = None):

        self.max_option = max_option
        self.children: list[DTBranch] = []
        self._candidate_option_branch = initial_node

        self.split_features = set()
        self._bestG = 0

    def add_option(self, x: DTBranch, split_decision):
        self.children.append(x)  # add decision branch as Option children

        self.split_features.add(split_decision.feature)  # update the used features in Option Node
        self._candidate_option_branch = None  # reset the candidate Option Branch

        if self._bestG < split_decision.merit:  # update the best G
            self._bestG = split_decision.merit

        if len(self.children) == self.max_option:  # clean statistics
            delattr(self, "split_features")

    def has_children(self):
        return len(self.children) > 0

    def can_split(self):
        return len(self.children) < self.max_option

    def has_candidate_option_branch(self):
        return self._candidate_option_branch is not None

    @property
    def bestG(self):
        return self._bestG

    @property
    def candidate_option_branch(self):
        return self._candidate_option_branch

    @candidate_option_branch.setter
    def candidate_option_branch(self, value):
        self._candidate_option_branch = value

    def can_add_candidate(self):
        return self._candidate_option_branch is None and self.max_option > len(self.children) > 0
