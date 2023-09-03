import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


from .tree import Tree


class DisconnectivityTree:
    def __init__(self, tree, de, graph):
        self.tree = tree
        self.de = de
        self.graph = graph

    def plot(self):
        tree = self.tree
        de = self.de

        global_minimum = sorted([n for n in self.graph.nodes()], key=lambda x: x.energy)[0]

        xlayout = recurse_xlayout(tree, global_minimum)
        line_segments = []
        recurse_line_segments(tree, xlayout, de, line_segments)

        _, ax = plt.subplots(figsize=(6, 7))

        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')

        leaves = tree.get_leaves()
        energies = [l.minimum.energy for l in leaves]
        xpos = [xlayout[l] for l in leaves]
        ax.plot(xpos, energies, '.', color='k')

        linecollection = LineCollection([[(x[0], y[0]), (x[1], y[1])] for x, y in line_segments])
        linecollection.set_color('k')
        ax.add_collection(linecollection)

        ax.set_xticks([])
        ax.autoscale_view(scalex=True, scaley=True, tight=False)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        plt.show()


def make_disconnectivity_tree(graph, levels=4):
    transition_states = sorted([ts.get('ts') for (m1, m2, ts) in graph.edges(data=True)], key=lambda x: -x.energy)
    elist = [x.energy for x in transition_states]

    if max(elist) > min(elist) + 1e-10:
        de = (max(elist) - min(elist)) / (levels - 1)
    else:
        de = 1.  # original code didn't consider this case

    # the upper edge of the bins
    # original code did not have +1, then last ts is neglected
    energy_levels = [min(elist) + de * i for i in range(levels + 1)]
    trees = iterate_through_levels_to_make_trees(transition_states, energy_levels)
    if len(trees) == 1:
        tree = trees[0]
    else:
        tree = Tree(
            minimum=None,
            ilevel=len(energy_levels) - 1,
            ethresh=energy_levels[-1] + 1. * de
        )
        for t in trees:
            tree.add_branch(t)

    dtree = DisconnectivityTree(tree, de, graph)
    return dtree


def iterate_through_levels_to_make_trees(transition_states, energy_levels):
    trees = []
    # union_find keeps track of connected components
    union_find = nx.utils.UnionFind()
    for i, ethresh in enumerate(energy_levels):

        # Get all minima below current energy level
        # Add them to current tree list
        while len(transition_states) > 0:
            ts = transition_states[-1]

            if ts.energy >= ethresh:
                break

            new_minima = []
            min1, min2 = ts.minimum1, ts.minimum2
            if min1 not in union_find.parents:
                new_minima.append(min1)
            if min2 not in union_find.parents:
                new_minima.append(min2)

            # add minima to cluster
            union_find.union(min1, min2)

            for m in new_minima:
                trees.append(Tree(m, None, None))

            transition_states.pop()

        # Add discrete transition state as parent for each connected cluster to new tree list
        new_trees = []
        cluster_to_parent_tree = {}
        connected_clusters = list(c1 for c1, c2 in union_find.parents.items() if c1 == c2)
        for cluster in connected_clusters:
            new_tree = Tree(None, i, ethresh)
            new_trees.append(new_tree)
            cluster_to_parent_tree[cluster] = new_tree

        # Get parent of clusters and add tree as subtree
        for tree in trees:
            # leaf
            if len(tree.subtrees) == 0:
                m = tree.minimum
            # no leaf
            elif tree.random_minimum != None:
                m = tree.random_minimum
            else:
                subtree = tree.subtrees[0]
                m = subtree.get_one_minimum()
                tree.random_minimum = m

            cluster = union_find[m]
            parent = cluster_to_parent_tree[cluster]

            if len(tree.subtrees) == 1:
                subtree = next(iter(tree.subtrees))
                parent.add_branch(subtree)
            else:
                parent.add_branch(tree)

        trees = new_trees

    return trees


def order_trees(tree_list, global_minimum):
    tree_value_list = [(tree.number_of_leaves(), tree) for tree in tree_list]

    for i, (v, tree) in enumerate(tree_value_list):
        if global_minimum in [leaf.minimum for leaf in tree.get_leaves()]:
            tree_value_list[i] = (-1, tree_value_list[i][1])
            break

    tree_value_list = sorted(tree_value_list, key=lambda x: x[0])
    res = []
    for i, (_, t) in enumerate(tree_value_list):
        if i % 2 == 0:
            res.append(t)
        else:
            res = [t] + res

    return res


def recurse_xlayout(tree, global_minimum, xmin=4, dx_xmin=1, xlayout={}):
    x = xmin + dx_xmin * tree.number_of_leaves() / 2.
    xlayout[tree] = x

    xmin_sub = xmin
    subtrees = order_trees(tree.subtrees, global_minimum)
    for subtree in subtrees:
        recurse_xlayout(subtree, global_minimum, xmin_sub)
        xmin_sub += subtree.number_of_leaves() * dx_xmin

    return xlayout


def recurse_line_segments(tree, xlayout, dy, line_segments=[]):
    if tree.parent is None:
        x = xlayout[tree]
        y = tree.ethresh
        line_segments.append(([x, x], [y, y + dy]))
    else:
        xparent = xlayout[tree.parent]
        xself = xlayout[tree]
        yparent = tree.parent.ethresh

        if len(tree.subtrees) == 0:
            yself = tree.minimum.energy
        else:
            yself = tree.ethresh

        # determine yhigh from eoffset
        yhigh = yparent - dy
        if yhigh <= yself:
            draw_vertical = False
            yhigh = yself
        else:
            draw_vertical = True

        # draw vertical line
        if len(tree.subtrees) == 0 and not draw_vertical:
            pass
        else:
            # add vertical line segment
            line_segments.append(([xself, xself], [yself, yhigh]))

        # draw the diagonal line
        line_segments.append(([xself, xparent], [yhigh, yparent]))

    for subtree in tree.subtrees:
        recurse_line_segments(subtree, xlayout, dy, line_segments)
