import tensorflow as tf
from memory import Memory
from ntm_cell import NTMCell


class NTM(object):
    def __init__(self, cell, mem_size, mem_dim, unit_num, seq_len, batch_size):
        self.memory = Memory(mem_dim=mem_dim, mem_size=mem_size, batch_size=batch_size)
        self.unit_num = unit_num
        self.seq_len = seq_len
        self.cell = cell

    def _step_op(self):
        pass