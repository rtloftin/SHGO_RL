class Tree(object):
    def __init__(self, minimum, ilevel, ethresh):
        self.minimum = minimum
        self.ilevel = ilevel
        self.ethresh = ethresh

        self.random_minimum = None
        self.subtrees = []
        self.parent = None

    def get_one_minimum(self):
        if len(self.subtrees) == 0:
            return self.minimum
        elif self.random_minimum != None:
            return self.random_minimum
        else:
            m = self.subtrees[0].get_one_minimum()
            self.random_minimum = m
            return m

    def number_of_leaves(self):
        """return the number of leaves that are descendants of this Tree"""
        if len(self.subtrees) == 0:
            nleaves = 1
        else:
            nleaves = 0
            for tree in self.subtrees:
                nleaves += tree.number_of_leaves()
        return nleaves

    def get_leaves(self):
        """return a list of the leaves that are descendants of this Tree"""
        if len(self.subtrees) == 0:
            leaves = [self]
        else:
            leaves = []
            for tree in self.subtrees:
                leaves += tree.get_leaves()
        return leaves

    def add_branch(self, branch):
        """make branch a child of this tree"""
        self.subtrees.append(branch)
        branch.parent = self