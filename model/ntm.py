import tensorflow as tf
from ntm_cell import *
from ntm_ops import linear


class NTM(object):
    def __init__(self, cell, seq_len, input_dim, batch_size, lr):
        self.mem_size = cell.mem_size
        self.mem_dim = cell.mem_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.cell = cell
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, input_dim + 1), name="inputs")
        self.targets = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, input_dim + 1), name="targets")
        self.start = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_dim), name="start_symbol")
        self.w, self.b = self.build_params()
        self.outputs, self.final_state = self.build_graph()
        self.loss = self.build_loss()
        self.params = tf.trainable_variables()
        self.opt_op = self.optimizer(lr)

    def init_state(self):
        state = self.cell.zero_state(self.batch_size, tf.float32)
        h, c, w_w, r_w, m = state
        with tf.variable_scope("init_mem"):
            m = linear(tf.reshape(m, (-1, self.mem_dim * self.mem_size)), output_size=self.mem_dim * self.mem_size,
                       bias=True, scope="mem")
            m = tf.reshape(m, (-1, self.mem_size, self.mem_dim))

        with tf.variable_scope("init_read_w"):
            r_w = linear(r_w, output_size=self.mem_size, bias=True, scope="r_w")

        with tf.variable_scope("init_write_w"):
            w_w = linear(w_w, output_size=self.mem_size, bias=True, scope="w_w")

        return NTMStateTuple(h, c, w_w, r_w, m)

    def build_params(self):
        with tf.variable_scope("ntm"):
            w = tf.get_variable(name="w", initializer=tf.random_normal_initializer(stddev=0.5),
                                shape=(self.cell.unit_num, self.input_dim + 1))
            b = tf.get_variable(name='b', initializer=tf.constant_initializer(0), shape=self.input_dim + 1)
        return w, b

    def build_graph(self):
        with tf.variable_scope("ntm"):
            input_seq = self._split_seq(self.inputs)
            # hiddens, state = tf.nn.rnn(self.cell, input_seq, initial_state=self.initial_state, scope="ntm_cell")
            hiddens = []
            state = self.init_state()
            for seq in range(0, self.seq_len):
                print "Building Graph %s/%s" % (seq + 1, self.seq_len)
                x = input_seq[seq]
                if seq > 0:
                    tf.get_variable_scope().reuse_variables()

                hidden, state = self.cell(x, state)
                hiddens.append(hidden)

            outputs = []
            for hidden in hiddens:
                output = tf.matmul(hidden, self.w) + self.b
                # output = tf.nn.sigmoid(output)
                outputs.append(output)
            # outputs = []
            # # decoder
            # with tf.variable_scope("ntm_cell", reuse=True):
            #     output, state = self.cell(self.start, state)
            #     output = build_output_layer(output)
            #     for i in range(0, self.seq_len):
            #         print "Building graph %s/%s" % (i, self.seq_len)
            #         output, state = self.cell(output, state)
            #         output = build_output_layer(output)
            #         outputs.append(output)
        return tf.transpose(tf.convert_to_tensor(outputs), perm=(1, 0, 2)), state

    def build_loss(self):
        with tf.variable_scope("ntm_loss"):
            outputs = self._split_seq(self.outputs)
            targets = self._split_seq(self.targets)
            losses = []
            for index in range(0, self.seq_len):
                print "Building loss %s/%s" % (index, self.seq_len)
                # loss for each output - target pair. has shape (batch_size, input_dim)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(outputs[index], targets[index])
                losses.append(tf.reduce_mean(tf.reduce_sum(loss, reduction_indices=1)))
        return tf.reduce_sum(tf.convert_to_tensor(losses))

    def optimizer(self, lr):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        gvs = optimizer.compute_gradients(self.loss, var_list=self.params)
        capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
        return optimizer.apply_gradients(capped_gvs)

    def _split_seq(self, x):
        """
        split x into a seq_len list with each element a 2D Tensor (batch_size, input_dim)

        Parameters:
        -----------
        x: 3D Tensor with shape (batch_size, seq_len, input_dim)

        Returns: A seq_len list with each element a 2D Tenosr
        """
        x = tf.transpose(x, perm=(1, 0, 2))
        x = tf.reshape(x, shape=(-1, self.input_dim + 1))
        x = tf.split(num_split=self.seq_len, split_dim=0, value=x)
        return x