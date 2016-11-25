from ntm_ops import *
import tensorflow as tf


class Memory(object):
    def __init__(self, mem_size=128, mem_dim=20, batch_size=1):
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
        self.batch_size = batch_size
        with tf.name_scope("external_memory"):
            self.memory = tf.fill((batch_size, mem_size, mem_dim), 1e-6, name="memory")

            # initialize read and write weighting with small values. multiple heads may be added later
            self.read_weighting = tf.fill((batch_size, mem_size), value=1e-6, name="read_weighting")
            self.write_weighting = tf.fill((batch_size, mem_size), value=1e-6, name="write_weighting")

            # initialize read vector
            self.read_vector = tf.fill((batch_size, mem_dim), value=1e-6, name="read_vector")

    def update_weighting(self, key_vector, key_strength, interpolation_factor, shift_weighting, sharpen_factor,
                         weighting, memory):
        """
        Parameters:
        -----------
        key_vector: Tensor (mem_size)
            Key vector emitted by the controller, same shape of memory dimension.
        key_strength: Tensor (scalar)
            Key strength emitted by the controller, a scalar.
        interpolation_factor:
        shift_weighting:
        sharpen_factor:
        weighting: bool
        memory:

        :return:
        """
        # 3.3.1
        content_based_weighting = self.content_addressing(key_vector, key_strength, memory)

        # 3.3.2
        w = self.location_addressing(content_based_weighting, interpolation_factor, shift_weighting,
                                     sharpen_factor, weighting)

        return w

    def content_addressing(self, key_vector, key_strength, memory):
        """
        content based addressing, # 3.3.1

        Parameters:
        -----------
        key_vector: Tensor (batch_size, mem_dim)
            Key vector emitted by the controller, same shape of memory dimension.
        key_strength: Tensor (batch_size, 1)
            Key strength emitted by the controller, a scalar.
        memory: Tensor (batch_size, mem_size, mem_dim)
            The memory matrix

        Returns: Tensor (batch_size, mem_size)
            The weighting of content based addressing.
        """
        with tf.name_scope("content_based_addressing"):
            # 1. compare similarity and apply key strength
            similarity = cosine_similarity(key_vector, memory) * key_strength
            # 2. normalized weighting
            w = tf.nn.softmax(similarity, name="content_weighting")
        return w

    def location_addressing(self, content_based_weighting, interpolation_factor, shift_weighting,
                            sharpen_factor, weighting):
        """
        location based addressing, # 3.3.2

        Parameters:
        -----------
        content_based_weighting: Tensor (batch_size, mem_size)
            the weighting after content based addressing
        interpolation_factor: Tensor (batch_size, 1)
            the scalar used to blend previous weighting and current weighting
        shift_weighting: Tensor (batch_size, allow_shift_dim)
            defines allowed integer shift using circular convolution
        sharpen_factor: Tensor (batch_size, 1)
            used to sharpen the weighting, let it mainly focus on one location
        weighting: Tensor (batch_size, mem_size)
            indicate which tensor to return

        Returns: 2 Tensors
            read_weighting, write_weighting
        """
        with tf.name_scope("location_based_addressing"):
            # 1. interpolation
            new_weighting = interpolation_factor * content_based_weighting + (tf.ones_like(weighting) -
                                                                              interpolation_factor) * weighting

            # 2. shifting
            new_weighting = circular_convolution(new_weighting, shift_weighting)

            # 3. sharpening
            new_weighting = tf.pow(new_weighting, sharpen_factor)
            new_weighting = tf.nn.softmax(new_weighting)

        return new_weighting

    def read(self, read_weighting, memory):
        """
        Read operation from the memory, # 3.1

        Parameters:
        -----------
        read_weighting:  Tensor (batch_size, mem_size)
            The read weighting at time step t

        Returns: Tensor (batch_size, mem_dim)

        """
        with tf.name_scope("read"):
            weighting = tf.reshape(read_weighting, shape=(self.batch_size, self.mem_size, 1))
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