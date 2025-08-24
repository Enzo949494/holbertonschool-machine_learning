#!/usr/bin/env python3
"""
Isolation Forest implementation for outlier detection.
This module provides an Isolation Random Forest for anomaly detection.
"""

import numpy as np

Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    Isolation Random Forest for outlier detection.
    
    An ensemble of Isolation Random Trees that identifies outliers
    by averaging the isolation depths across multiple trees.
    """
    
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predict the average isolation depth for each sample.
        
        Args:
            explanatory (numpy.ndarray): Input features to predict.
            
        Returns:
            numpy.ndarray: Average isolation depths for each sample.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fit the Isolation Forest to the data.
        
        Args:
            explanatory (numpy.ndarray): Input features for training.
            n_trees (int, optional): Number of trees to create. Defaults to 100.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth, 
                                    seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory that have the smallest 
        mean depth.
        
        Args:
            explanatory (numpy.ndarray): Input features to analyze.
            n_suspects (int): Number of suspects to return.
            
        Returns:
            tuple: (suspects_data, suspects_depths) where suspects_data 
                   contains the n_suspects samples with lowest depths and
                   suspects_depths contains their corresponding depths.
        """
        depths = self.predict(explanatory)
        
        # Get indices of the n_suspects smallest depths
        suspect_indices = np.argsort(depths)[:n_suspects]
        
        # Get the corresponding data points and their depths
        suspects_data = explanatory[suspect_indices]
        suspects_depths = depths[suspect_indices]
        
        return suspects_data, suspects_depths
