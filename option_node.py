from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch

class OptionNode:

    _traverse_mode = ["first", "all"]

    def __init__(self):
        self.children = []
        # TODO - capire come fare in modo di salvarsi gli attributi dei parents
        self.used_attributes = []

    def add_option_branch(self, x: DTBranch | HTLeaf) -> None:
        self.children.append(x)

    def traverse(self, mode="first") -> DTBranch | HTLeaf:
        if mode not in self._traverse_mode:
            print("Invalid traverse mode. Using default.")
            mode = "first"
        # TODO - fare in modo che scenda in parallelo su tutti gli option branch
        if mode == "first":
            return self.children[0]
        else:
            pass