from model.ntm import NTM
from model.ntm_cell import *
import tensorflow as tf

mem_size, mem_dim = 128, 20
hidden_unit = 100
seq = 20
input_dim = 8
batch_size = 1

ntm_cell = NTMCell(hidden_unit, mem_dim=mem_dim, mem_size=mem_size)
ntm = NTM(batch_size=batch_size, cell=ntm_cell, input_dim=input_dim, seq_len=seq)

