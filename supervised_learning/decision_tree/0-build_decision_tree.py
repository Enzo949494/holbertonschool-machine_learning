#!/usr/bin/env python3
"""
Decision tree build module for machine learning.
This module provides classes to build and manage decision trees.
"""


import numpy as np


class Node:
    """
    A node in a decision tree.

    Represents an internal node that makes decisions based on
    feature values and thresholds.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a Node.

        Args:
            feature: Index of the feature used for splitting
            threshold: Threshold value for the feature
            left_child: Left child node
            right_child: Right child node
            is_root: Boolean indicating if this is the root node
            depth: Depth of this node in the tree
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculate the maximum depth of the subtree rooted at this node.

        Returns:
            int: Maximum depth of the subtree
        """
        if self.left_child is None and self.right_child is None:
            return self.depth

        max_depth = self.depth

        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        return max_depth


class Leaf(Node):
    """
    A leaf node in a decision tree.

    Represents a terminal node that contains a prediction value.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf node.

        Args:
            value: The prediction value of this leaf
            depth: Depth of this leaf in the tree
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf node.

        Returns:
            int: Depth of this leaf node
        """
        return self.depth


class Decision_Tree():
    """
    A decision tree classifier.

    Implements a decision tree for classification tasks.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision Tree.

        Args:
            max_depth: Maximum depth of the tree
            min_pop: Minimum population required to split a node
            seed: Random seed for reproducibility
            split_criterion: Criterion used for splitting nodes
            root: Root node of the tree
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Calculate the maximum depth of the decision tree.

        Returns:
            int: Maximum depth of the tree
        """
        return self.root.max_depth_below()
