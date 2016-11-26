from ntm_ops import linear
import tensorflow as tf
import collections


NTMStateTuple = collections.namedtuple("NTMStateTuple", ("h", "c", "m"))


class NTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, unit_num, mem_size, mem_dim, activation=tf.nn.tanh, shift_weighting=3, forget_bias=1.0):
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
            c, h, memory = state
            memory_matrix = memory.memory
            write_weighting = memory.write_weighting
            read_weighting = memory.read_weighting
            read_vector = memory.read_vector
            # one step of lstm
            with tf.variable_scope("lstm"):
                # res has shape (batch_size, unit_num * 4)
                res = linear([c, h, read_vector], self.unit_num * 4, bias=True, bias_start=0.1)

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
                k = tf.nn.relu(linear(h, output_size=self.mem_dim, bias=False, scope="key_vector"))
                beta = tf.nn.relu(linear(h, output_size=1, bias=False, scope="key_strength"))
                g = tf.nn.sigmoid(linear(h, output_size=1, bias=False, scope="interpolation"))
                s = tf.nn.softmax(linear(h, output_size=self.shift_weighting, bias=False, scope="shifting"))
                gamma = tf.nn.relu(linear(h, output_size=1, bias=False, scope="sharpening")) + 1

            # write to memory
            with tf.variable_scope("write"):
                # erase vector
                erase = tf.nn.sigmoid(linear(h, output_size=self.mem_dim, bias=False, scope="erase"))

                # add vector
                add = tf.nn.tanh(linear(h, output_size=self.mem_dim, bias=False, scope="add"))

                # write
                memory_matrix = memory.write(write_weighting, memory_matrix, erase_vector=erase, add_vector=add)

                # update write weighting
                write_weighting = memory.update_weighting(k, beta, g, s, gamma, write_weighting, memory_matrix)

            # read from memory
            with tf.variable_scope("read"):
                # read
                read_vector = memory.read(read_weighting, memory_matrix)

                # update read weighting
                read_weighting = memory.update_weighting(k, beta, g, s, gamma, read_weighting, memory_matrix)

            # update memory
            memory.memory = memory_matrix
            memory.write_weighting = write_weighting
            memory.read_weighting = read_weighting
            memory.read_vector = read_vector

            state = NTMStateTuple(c, h, memory)
            return h, state

    @property
    def state_size(self):
        return NTMStateTuple(self.unit_num, self.unit_num, (self.mem_size, self.mem_dim))

    @property
    def output_size(self):
        return self.unit_num

    def zero_state(self, batch_size, dtype):

        return 1