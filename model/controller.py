import tensorflow as tf
import collections


_NTMStateTuple = collections.namedtuple("NTMStateTuple", ("h", "c", "r"))


class NTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, unit_num, seq_len, read_len, stacked_layer=1):
        """

        unit_num:
        seq_len:
        read_len:
        stacked_layer:
        """
        self.unit_num = unit_num
        self.seq_len = seq_len
        self.stacked_layer = stacked_layer
        self.read_len = read_len

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
            A pair containing: (output(hidden_state), state(hidden_state, cell_state, read_vector))
        """

    @property
    def state_size(self):
        return _NTMStateTuple(self.unit_num, self.unit_num, self.read_len)

    @property
    def output_size(self):
        return self.unit_num
