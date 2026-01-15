#!/usr/bin/env python3
"""
Bidirectional RNN Cell with Output Layer

This module implements a bidirectional recurrent neural network cell that processes
sequences in both forward and backward directions, then combines the hidden states
to produce outputs.
"""

import numpy as np


class BidirectionalCell:
    """
    Bidirectional RNN cell that processes data in both forward and backward directions.
    
    This cell maintains separate hidden state dimensions for forward and backward
    passes, and combines them to generate outputs using a shared output layer.
    
    Attributes:
        Whf (np.ndarray): Weight matrix for forward hidden state of shape (i+h, h)
        Whb (np.ndarray): Weight matrix for backward hidden state of shape (i+h, h)
        Wy (np.ndarray): Weight matrix for output layer of shape (2*h, o)
        bhf (np.ndarray): Bias vector for forward hidden state of shape (1, h)
        bhb (np.ndarray): Bias vector for backward hidden state of shape (1, h)
        by (np.ndarray): Bias vector for output layer of shape (1, o)
    """

    def __init__(self, i, h, o):
        """
        Initialize a bidirectional RNN cell.
        
        Args:
            i (int): Dimensionality of the input data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the output
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculate the hidden state in the forward direction for one time step.
        
        Args:
            h_prev (np.ndarray): Previous hidden state of shape (m, h)
            x_t (np.ndarray): Input data at current time step of shape (m, i)
            
        Returns:
            np.ndarray: Next hidden state of shape (m, h)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculate the hidden state in the backward direction for one time step.
        
        Args:
            h_next (np.ndarray): Next hidden state (in backward pass) of shape (m, h)
            x_t (np.ndarray): Input data at current time step of shape (m, i)
            
        Returns:
            np.ndarray: Previous hidden state (backward direction) of shape (m, h)
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculate outputs for all time steps using concatenated hidden states.
        
        Applies an affine transformation followed by softmax activation to convert
        the concatenated forward and backward hidden states into probability distributions.
        
        Args:
            H (np.ndarray): Concatenated hidden states of shape (t, m, 2*h)
                           where t is the number of time steps, m is batch size,
                           and 2*h combines forward and backward hidden dimensions
            
        Returns:
            np.ndarray: Output probabilities after softmax of shape (t, m, o)
                       where o is the output dimensionality
        """
        t, m, _ = H.shape
        o = self.Wy.shape[1]

        # Affine transformation: (t, m, 2h) -> (t, m, o)
        Y_lin = H @ self.Wy + self.by  # broadcasting by over (t, m)

        # Softmax over last dimension
        exp_Y = np.exp(Y_lin)
        Y = exp_Y / np.sum(exp_Y, axis=2, keepdims=True)

        return Y
