from ntm_ops import linear
import tensorflow as tf
import collections
from addressing import *
import numpy as np
from ntm_ops import *


NTMStateTuple = collections.namedtuple("NTMStateTuple", ("h", "c", "write_w", "read_w", "mem"))


class NTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, unit_num, mem_size, mem_dim,
                 activation=tf.nn.tanh, shift_weighting=3, forget_bias=1.0):
        """

        unit_num:
        seq_len:
        read_len:
        """
        self.unit_num = unit_num
        self._forget_bias = forget_bias
        self._activation = activation
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.shift_weighting = shift_weighting

    def __call__(self, inputs, state, scope=None):
        """
        Run NTM cell given the inputs and state

        Parameters:
        -----------
        inputs: Tensor (batch_size, input_size)
            inputs of one time step
        state: A tuple of three tensors (hidden_state, cell_state, read_vector)
            state of current time step
        scope: name of this NTM cell

        Returns:
            A pair containing: (output(hidden_state), state(hidden_state, cell_state, read_vector, memory))
        """
        with tf.variable_scope("ntm_cell" if scope is None else scope):
            c, h, write_weighting, read_weighting, memory_matrix = state

            # write to memory
            with tf.variable_scope("write"):
                # erase vector
                erase = tf.nn.sigmoid(linear(h, output_size=self.mem_dim, bias=True, scope="erase"))

                # add vector
                add = tf.nn.tanh(linear(h, output_size=self.mem_dim, bias=True, scope="add"))

                # write, update memory
                memory_matrix = self.write(write_weighting, memory_matrix, erase_vector=erase, add_vector=add)

            with tf.variable_scope("read"):
                # read
                read_vector = self.read(read_weighting, memory_matrix)

            with tf.variable_scope("lstm"):
                # res has shape (batch_size, unit_num * 4)
                res = linear([c, h, read_vector, inputs], self.unit_num * 4, bias=True, bias_start=0.1)

                # split res into four gates: input, new_input, forget, output
                i, j, f, o = tf.split(1, 4, res)

                # forget gate
                f = tf.nn.sigmoid(f + self._forget_bias, "forget")

                # input gate
                i = tf.nn.sigmoid(i, "input")

                # new input gate (select candidate cell state)
                j = self._activation(j, "new_input")

                # output gate
                o = tf.nn.sigmoid(o, "output")

                # compute new cell state
                c = f * c + (i * j)
                # compute new hidden state
                h = o * self._activation(c)

            # addressing
            with tf.variable_scope("addressing"):
                with tf.variable_scope("writing"):
                    k_w = tf.nn.tanh(linear(h, output_size=self.mem_dim, bias=True, scope="key_vector"), "key_vector")
                    beta_w = tf.nn.softplus(linear(h, output_size=1, bias=True, scope="key_strength"), "beta")
                    g_w = tf.nn.sigmoid(linear(h, output_size=1, bias=True, scope="interpolation"), "gate")
                    s_w = tf.nn.softmax(linear(h, output_size=self.shift_weighting, bias=True, scope="shifting"), "shifting")
                    gamma_w = tf.nn.softplus(linear(h, output_size=1, bias=True, scope="sharpening"), "gamma") + tf.constant(1.0)

                with tf.variable_scope("reading"):
                    k_r = tf.nn.tanh(linear(h, output_size=self.mem_dim, bias=True, scope="key_vector"), "key_vector")
                    beta_r = tf.nn.softplus(linear(h, output_size=1, bias=True, scope="key_strength"), "beta")
                    g_r = tf.nn.sigmoid(linear(h, output_size=1, bias=True, scope="interpolation"))
                    s_r = tf.nn.softmax(linear(h, output_size=self.shift_weighting, bias=True, scope="shifting"), "shifting")
                    gamma_r = tf.nn.softplus(linear(h, output_size=1, bias=True, scope="sharpening"), "gamma") + tf.constant(1.0)

            # update weights
            write_weighting = update_weighting(k_w, beta_w, g_w, s_w, gamma_w, write_weighting, memory_matrix)
            read_weighting = update_weighting(k_r, beta_r, g_r, s_r, gamma_r, read_weighting, memory_matrix)

            # update state
            state = NTMStateTuple(c, h, write_weighting, read_weighting, memory_matrix)
            return h, state

    def read(self, read_weighting, memory):
        """
        Read operation from the memory, # 3.1

        Parameters:
        -----------
        read_weighting:  Tensor (batch_size, mem_size)
            The read weighting at time step t

        Returns: Tensor (batch_size, mem_dim)

        """
        batch_size = memory.get_shape().as_list()[0]
        weighting = tf.reshape(read_weighting, shape=(batch_size, self.mem_size, 1))
        read_vector = tf.mul(memory, weighting)
        read_vector = tf.reduce_sum(read_vector, reduction_indices=1)
        return read_vector

    def write(self, write_weighting, memory, erase_vector, add_vector):
        """
        Parameters:
        -----------
        write_weighting: Tensor (batch_size, mem_size)
            the write weighting of the current time step
        memory: Tensor(batch_size, mem_size, mem_dim)
            the memory from previous time step
        erase_vector: Tensor(batch_size, mem_dim)
            the erase vector
        add_vector: Tensor(batch_size, mem_dim)
            the add vector

        Returns: Tensor shape same as memory
        """
        # erase
        erase = linear_combination(write_weighting, erase_vector)
        erase = tf.ones_like(erase, dtype=tf.float32) - erase
        memory = tf.mul(memory, erase)
        # add
        add = linear_combination(write_weighting, add_vector)
        memory = memory + add
        return memory

    @property
    def state_size(self):
        return NTMStateTuple(self.unit_num, self.unit_num, (self.mem_size, self.mem_dim))

    @property
    def output_size(self):
        return self.unit_num

    def zero_state(self, batch_size, dtype):
        memory_matrix = tf.fill((batch_size, self.mem_size, self.mem_dim), value=1e-6, name="memory")
        write_w = tf.fill((batch_size, self.mem_size), value=1e-6, name="write_w")
        read_w = tf.fill((batch_size, self.mem_size), value=1e-6, name="read_weighting")
        c = tf.zeros(shape=(batch_size, self.unit_num), dtype=dtype, name="init_cell_state")
        h = tf.zeros(shape=(batch_size, self.unit_num), dtype=dtype, name="init_hidden_state")
        return NTMStateTuple(h, c, write_w, read_w, memory_matrix)
