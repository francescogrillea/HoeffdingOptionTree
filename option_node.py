from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch


class OptionNode:
    _traverse_mode = ["first", "all"]

    def __init__(self, initial_node: DTBranch = None):
        if initial_node is None:
            self.children = []  # TODO - valutare se metterla private
        else:
            self.children = [initial_node]
        self.used_attributes = []        # TODO - capire come fare in modo di salvarsi gli attributi dei parents
        self.bestG = 0  # TODO - trasformarlo in un property?

    def add_option_branch(self, x: DTBranch | HTLeaf):
        self.children.append(x)

    def has_children(self):
        return len(self.children) > 0

    def option_count(self):
        return len(self.children)

    def traverse(self, mode="first"):
        # TODO - controllare se va bene passarla per riferimento
        return self.children

    def __repr__(self):
        return f"<OptionNode: {[c.__class__.__name__ for c in self.children]}>"