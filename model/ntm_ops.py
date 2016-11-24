import tensorflow as tf


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


def circular_convolution(weighting, shift_weighting):
    """
    Circular convolution shift

    Parameter:
    ----------
    weighting: Tensor (batch_size, mem_size)
        the weighting for shifting
    shift_weighting: Tensor (batch_size, allowed_int_shift)

    Returns: Tensor (same shape as weighting)
    """
