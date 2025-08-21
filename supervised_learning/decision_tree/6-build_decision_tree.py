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

    def update_bounds_below(self):
        """
        Update bounds for this node and all nodes below.

        Calculates lower and upper bounds for each feature based on
        the decision tree structure.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                # Initialize child bounds with parent bounds
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                # Update bounds based on the split
                if child == self.left_child:
                    # Left child: feature > threshold
                    child.lower[self.feature] = self.threshold
                else:
                    # Right child: feature <= threshold
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

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

    def update_indicator(self):
        """
        Update the indicator function for this node.

        Creates a function that determines if individuals satisfy
        the node's conditions based on lower and upper bounds.
        """

        def is_large_enough(x):
            """
            Check if individuals meet lower bound conditions.

            Args:
                x: 2D array of shape (n_individuals, n_features)

            Returns:
                1D boolean array indicating if each individual
                has all features > lower bounds
            """
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.lower.keys():
                if self.lower[key] == -np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(
                        np.greater(x[:, key], self.lower[key]))

            return np.all(np.array(conditions), axis=0)

        def is_small_enough(x):
            """
            Check if individuals meet upper bound conditions.

            Args:
                x: 2D array of shape (n_individuals, n_features)

            Returns:
                1D boolean array indicating if each individual
                has all features <= upper bounds
            """
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.upper.keys():
                if self.upper[key] == np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(np.less_equal(x[:, key],
                                                    self.upper[key]))

            return np.all(np.array(conditions), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predict class for a single sample using this node.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def update_bounds_below(self):
        """
        Update bounds for leaf node (no operation needed).
        """
        pass

    def __str__(self):
        """
        String representation of the leaf.

        Returns:
            str: Leaf representation
        """
        return f"-> leaf [value={self.value}]"

    def update_indicator(self):
        """
        Update the indicator function for this leaf.

        Creates a function that determines if individuals satisfy
        the leaf's conditions based on lower and upper bounds.
        """
        def is_large_enough(x):
            """
            Check if individuals meet lower bound conditions.
            """
            if not hasattr(self, 'lower') or not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.lower.keys():
                if self.lower[key] == -np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(
                        np.greater(x[:, key], self.lower[key]))

            return np.all(np.array(conditions), axis=0)

        def is_small_enough(x):
            """
            Check if individuals meet upper bound conditions.
            """
            if not hasattr(self, 'upper') or not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.upper.keys():
                if self.upper[key] == np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(
                        np.less_equal(x[:, key], self.upper[key]))

            return np.all(np.array(conditions), axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predict class for a single sample using this leaf.
        """
        return self.value


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

    def update_bounds(self):
        """
        Update bounds for all nodes in the tree.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Update the prediction function for the decision tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_function(A):
            predictions = np.zeros(A.shape[0])
            for leaf in leaves:
                mask = leaf.indicator(A)
                predictions[mask] = leaf.value
            return predictions.astype(int)

        self.predict = predict_function

    def pred(self, x):
        """
        Predict class for a single sample.
        """
        return self.root.pred(x)

    def __str__(self):
        """
        String representation of the tree.

        Returns:
            str: Tree structure as string
        """
        return self.root.__str__()
