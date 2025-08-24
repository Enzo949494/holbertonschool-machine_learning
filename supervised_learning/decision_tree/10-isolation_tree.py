#!/usr/bin/env python3
"""
Isolation Tree implementation for outlier detection.
This module provides an Isolation Random Tree for anomaly detection.
"""

import numpy as np

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Isolation Random Tree for outlier detection.

    This tree isolates outliers by randomly splitting data until each point
    is isolated. Outliers are expected to be isolated with fewer splits.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """String representation of the tree."""
        return self.root.__str__()

    def depth(self):
        """Get the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Update bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Get all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Update the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_function(A):
            predictions = np.zeros(A.shape[0])
            for leaf in leaves:
                mask = leaf.indicator(A)
                predictions[mask] = leaf.depth  # Return depth instead of value
            return predictions.astype(int)

        self.predict = predict_function

    def np_extrema(self, arr):
        """Get min and max of array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split criterion - same as Decision_Tree."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child - different from Decision_Tree."""
        # For isolation tree, leaf value is the depth (not a class)
        leaf_child = Leaf(node.depth + 1)  # Value = depth
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a node child - same as Decision_Tree."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Fit a single node."""
        node.feature, node.threshold = self.random_split_criterion(node)

        # Same as Decision_Tree
        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population

        # Different from Decision_Tree - only stop based on depth or population
        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Different from Decision_Tree - only stop based on depth or population
        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fit the isolation tree to the data.

        Args:
            explanatory (numpy.ndarray): Input features.
            verbose (int): Verbosity level.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
