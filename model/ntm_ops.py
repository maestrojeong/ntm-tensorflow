import tensorflow as tf


def cosine_similarity(key_vector, memory, eps=1e-8):
    """
    The cosine similarity between x1 and x2

    Parameters:
    -----------
    key_vector: Tensor (Rank of 2)
        the key vector to compare to the memory
    memory: Tensor (The last dimension is the same as x1)
        memory to be compared to
    eps: float32
        small float value to prevent divide by 0.

    Returns: Tensor (The first dimension is the same as x1 first dimension,
                     the second dimension is the same as x2 first dimension)
    """
    with tf.name_scope("cos_similarity"):
        # the numerator of cosine similarity
        numerator = tf.matmul(key_vector, memory, transpose_b=True)

        # the length of x1 tensor, shape(batch_size)
        x1_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(key_vector), reduction_indices=1)), shape=(-1, 1))

        # the length of x2 tensor, shape(mem_size)
        x2_len = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(memory), reduction_indices=1)), shape=(1, -1))

        denominator = tf.matmul(x1_len, x2_len)
        return numerator / (denominator + eps)


def circular_convolution(weighting, shift_weighting):
    """
    Parameter:
    ----------
    weighting: Tensor (batch_size, mem_size)
        the weighting for shifting
    shift_weighting: Tensor (batch_size, allowed_int_shift)

    Returns
    """