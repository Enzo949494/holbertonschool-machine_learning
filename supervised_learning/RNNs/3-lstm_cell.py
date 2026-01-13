#!/usr/bin/env python3
"""
LSTM cell
"""

import numpy as np


class LSTMCell:
    """Represents a cell of an LSTM unit."""

    def __init__(self, i, h, o):
        """
        i: dimension of data
        h: dimension of hidden state
        o: dimension of outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        h_prev: (m, h) previous hidden state
        c_prev: (m, h) previous cell state
        x_t:    (m, i) input at time t
        Returns: h_next, c_next, y
        """
        # concaténation (m, i + h)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        f_t = 1 / (1 + np.exp(-(concat @ self.Wf + self.bf)))

        # update gate (input gate)
        u_t = 1 / (1 + np.exp(-(concat @ self.Wu + self.bu)))

        # état candidat de la cellule
        c_tilde = np.tanh(concat @ self.Wc + self.bc)

        # nouveau cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # output gate
        o_t = 1 / (1 + np.exp(-(concat @ self.Wo + self.bo)))

        # nouveau hidden state
        h_next = o_t * np.tanh(c_next)

        # sortie softmax
        y_lin = h_next @ self.Wy + self.by
        exp_y = np.exp(y_lin)
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
