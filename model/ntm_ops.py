import tensorflow as tf
import math


def cosine_similarity(key_vector, memory, eps=1e-8):
    """
    The cosine similarity between x1 and x2

    Parameters:
    -----------
    key_vector: Tensor (batch_size, mem_dim)
        the key vector to compare to the memory
    memory: Tensor (batch_size, mem_size, mem_dim)
        memory to be compared to
    eps: float32
        small float value to prevent divide by 0.

    Returns: Tensor (batch_size, mem_dim)
    """
    with tf.name_scope("cos_similarity"):
        batch_size = key_vector.get_shape()[0]
        key_vector_list = tf.split(0, batch_size, key_vector)
        memory_list = tf.split(0, batch_size, memory)
        memory_list = [tf.squeeze(mem_ele, squeeze_dims=[0]) for mem_ele in memory_list]

        # the numerator of cosine similarity, has shape(batch_size, mem_size)
        numerator = [tf.reduce_sum(key_ele * mem_ele, reduction_indices=1)
                     for key_ele, mem_ele in zip(key_vector_list, memory_list)]
        numerator = tf.convert_to_tensor(numerator, dtype=tf.float32)

        # the length of key_vector tensor, shape(batch_size)
        key_vector_len = tf.sqrt(tf.reduce_sum(tf.square(key_vector), reduction_indices=1))

        # the length of memory tensor, shape(batch_size, mem_size)
        mem_tensor_len = tf.sqrt(tf.reduce_sum(tf.square(memory), reduction_indices=2))

        # denominator of cosine similarity, has shape(batch_size, mem_size)
        denominator = tf.reshape(key_vector_len, (-1, 1)) * mem_tensor_len
        return numerator / (denominator + eps)


def circular_convolution(weighting, kernel):
    """
    Circular convolution shift

    Parameter:
    ----------
    weighting: Tensor (batch_size, mem_size)
        the weighting for shifting
    kernel: Tensor (batch_size, allowed_int_shift)

    Returns: Tensor (same shape as weighting)
    """
    with tf.name_scope("circular_conv"):
        size = int(weighting.get_shape()[1])
        kernel_size = int(kernel.get_shape()[1])
        kernel_shift = int(math.floor(kernel_size / 2.0))

        def loop(idx):
            if idx < 0: return size + idx
            if idx >= size:
                return idx - size
            else:
                return idx

        kernels = []
        for i in xrange(size):
            # get indices for circular convolution
            indices = [loop(i + j) for j in xrange(kernel_shift, -kernel_shift - 1, -1)]

            # gather columns from weighting
            _weighting = gather_cols(weighting, indices)
            # calculate convolution operation
            kernels.append(tf.reduce_sum(kernel * _weighting, 1))
        kernels = tf.convert_to_tensor(tf.transpose(kernels))
    return kernels


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])


def linear_combination(multiplier, multiplicand):
    """
    Parameters:
    -----------
    multiplier: Tensor (batch_size, mem_size)
    multiplicand: Tensor (batch_size, mem_dim)

    Returns: Tensor (batch_size, mem_dim)
    """
    batch_size, mem_size = multiplier.get_shape()
    mem_dim = multiplicand.get_shape()[1]
    # slice into batches
    w_s, e_s = tf.split(split_dim=0, num_split=int(batch_size), value=multiplier), \
               tf.split(split_dim=0, num_split=int(batch_size), value=multiplicand)
    # change shape to compute matrix multiplication
    w_s = [tf.reshape(w, (int(mem_size), 1)) for w in w_s]
    e_s = [tf.reshape(e, (1, int(mem_dim))) for e in e_s]
    r_s = []
    for w, e in zip(w_s, e_s):
        r = tf.matmul(w, e)
        r_s.append(r)
    r_s = tf.convert_to_tensor(r_s)
    return r_s