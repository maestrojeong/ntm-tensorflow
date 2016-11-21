import tensorflow as tf
import numpy as np


class Memory(object):
    def __init__(self, mem_size=128, mem_dim=20, batch_size=64):
        """
        The dimension of the external memory

        Parameters:
        -----------
        mem_size: int
            Number of locations of this NTM.
        mem_dim: int
            Size of dimension of this memory.
        batch_size: int
            Size of the input data batch.
        """
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        with tf.variable_scope("external_memory"):
            self.memory = tf.fill((mem_size, mem_dim), 1e-6, name="memory")

            # initialize read and write weighting with small values. multiple heads may be added later
            self.read_weighting = tf.fill((batch_size, mem_size), value=1e-6, name="read_weighting")
            self.write_weighting = tf.fill((batch_size, mem_size), value=1e-6, name="write_weighting")

            # initialize read vector
            self.read_vector = tf.fill((batch_size, mem_dim), value=1e-6, name="read_vector")

    def update_weighting(self, k, b, g, s, y):
        """
        Parameters:
        -----------
        k:
        b:
        g:
        s:
        y:
        :return:
        """
        # 3.3.1
        content_based_weighting = self.content_addressing(k, b)

        # 3.3.2
        w = self.location_addressing(content_based_weighting, g, s, y)

        return w

    def content_addressing(self, k, b):
        """
        content based addressing, # 3.3.1

        Parameters:
        -----------
        k: Tensor (batch_size, mem_dim)
            Key vector emitted by the controller, same shape of memory dimension.
        b: Tensor (batch_size, scalar)
            Key strength emitted by the controller, a scalar.

        Returns: Tensor (batch_size, mem_size)
            The weighting of content based addressing.
        """


    def location_addressing(self, content_based_weighting, g, s, y):
        pass

    def read(self, read_weighting):
        pass

    def write(self, write_weighting, erase, add):
        pass
