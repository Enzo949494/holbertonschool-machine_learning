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

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes in the subtree rooted at this node.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: Number of nodes in the subtree
        """
        count = 0

        if not only_leaves:
            count = 1

        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def left_child_add_prefix(self, text):
        """
        Add prefix for left child in tree visualization.

        Args:
            text: Text representation of the left child

        Returns:
            str: Formatted text with left child prefix
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Add prefix for right child in tree visualization.

        Args:
            text: Text representation of the right child

        Returns:
            str: Formatted text with right child prefix
        """
        lines = text.split("\n")
        new_text = "    +---" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("        " + x) + "\n"
        return new_text

    def __str__(self):
        """
        String representation of the node and its subtree.

        Returns:
            str: Tree structure as string
        """
        if self.is_root:
            node_str = (f"root [feature={self.feature}, "
                        f"threshold={self.threshold}]")
        else:
            node_str = (f"> node [feature={self.feature}, "
                        f"threshold={self.threshold}]")

        result = node_str

        if self.left_child is not None:
            left_str = str(self.left_child)
            result += "\n" + self.left_child_add_prefix(left_str).rstrip()

        if self.right_child is not None:
            right_str = str(self.right_child)
            result += "\n" + self.right_child_add_prefix(right_str).rstrip()

        return result


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

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes in the subtree rooted at this leaf.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: Number of nodes (always 1 for a leaf)
        """
        return 1

    def __str__(self):
        """
        String representation of the leaf.

        Returns:
            str: Leaf representation
        """
        return f"-> leaf [value={self.value}]"


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

    def count_nodes(self, only_leaves=False):
        """
        Count the total number of nodes in the decision tree.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: Number of nodes in the tree
        """
        return self.root.count_nodes_below(only_leaves)

    def __str__(self):
        """
        String representation of the decision tree.

        Returns:
            str: Tree structure as string
        """
        return self.root.__str__()
