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
            feature: The feature index for splitting
            threshold: The threshold value for splitting
            left_child: Left child node
            right_child: Right child node
            is_root: Whether this is the root node
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
        Calculate the maximum depth below this node.

        Returns:
            int: Maximum depth below this node
        """
        max_depth = self.depth
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this node.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: Number of nodes below
        """
        count = 0 if only_leaves else 1

        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def get_leaves_below(self):
        """
        Get all leaves below this node.

        Returns:
            list: List of all leaf nodes below this node
        """
        leaves = []
        
        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())
            
        return leaves

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
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
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
            node_str = (f"-> node [feature={self.feature}, "
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

    def __init__(self, value, depth=0):
        """
        Initialize a Leaf.

        Args:
            value: The prediction value for this leaf
            depth: Depth of this leaf in the tree
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf.

        Returns:
            int: Depth of this leaf
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count this leaf as one node.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: 1 (this leaf)
        """
        return 1

    def get_leaves_below(self):
        """
        Return this leaf in a list.

        Returns:
            list: List containing only this leaf
        """
        return [self]

    def __str__(self):
        """
        String representation of the leaf.

        Returns:
            str: Leaf representation
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    A decision tree classifier.

    Manages the root node and provides tree-level operations.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision Tree.

        Args:
            max_depth: Maximum depth of the tree
            min_pop: Minimum population to split a node
            seed: Random seed for reproducibility
            split_criterion: Criterion for splitting nodes
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
        Get the depth of the tree.

        Returns:
            int: Maximum depth of the tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the tree.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            int: Number of nodes in the tree
        """
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        """
        Get all leaves in the tree.

        Returns:
            list: List of all leaf nodes in the tree
        """
        return self.root.get_leaves_below()

    def __str__(self):
        """
        String representation of the tree.

        Returns:
            str: Tree structure as string
        """
        return self.root.__str__()