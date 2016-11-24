import tensorflow as tf
from memory import Memory
from controller import Controller

memory = Memory()
controller = Controller(128)


def step_op(x, prev_state):
    """
    One step of NTM

    Parameters:
    -----------
    x: Tensor (batch_size, seq_len)
        one step of the input tensor
    prev_state: tuple of (cell_state, hidden_state)
        the previous state of NTM

    Returns: tuple of (cell_state, hidden_state) of controller
    """
    # get memory matrix
    memory_matrix = memory.memory

    # get read_vector
    curr_read_vector = memory.read_vector

    # get write weighting
    curr_write_weighting = memory.write_weighting

    # get read weighting
    curr_read_weighting = memory.read_weighting

    # one step of controller, outputs the five addressing elements
    next_hidden, key_vector, key_strength, interpolation_factor, shift_weighting, sharpen_factor, erase_vector, \
                                                        add_vector = controller.step(x, curr_read_vector, prev_state)

    # write to the memory
    next_memory_matrix = memory.write(curr_write_weighting, memory=memory_matrix, add_vector=add_vector,
                                      erase_vector=erase_vector)
    # update write weighting
    next_write_weighting = memory.update_weighting(key_vector, key_strength, interpolation_factor, shift_weighting,
                                                   sharpen_factor, curr_write_weighting, memory_matrix)

    # read from memory
    next_read_vector = memory.read(read_weighting=curr_read_weighting, memory=memory_matrix)

    # update read weighting
    next_read_weighting = memory.update_weighting(key_vector, key_strength, interpolation_factor, shift_weighting,
                                                  sharpen_factor, curr_read_weighting, memory_matrix)

    # update memory
    memory.memory = next_memory_matrix
    memory.read_vector = next_read_vector
    memory.write_weighting = next_write_weighting
    memory.read_weighting = next_read_weighting

    return next_hidden


def lstm_step(x, prev_state):
    return 1