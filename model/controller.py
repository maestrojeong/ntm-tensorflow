import tensorflow as tf


class Controller(object):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def step(self, x, read_vector, prev_state):
        pass