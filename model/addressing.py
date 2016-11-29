from ntm_ops import *
import tensorflow as tf


def update_weighting(key_vector, key_strength, interpolation_factor, shift_weighting, sharpen_factor,
                     weighting, memory):
    """
    Parameters:
    -----------
    key_vector: Tensor (batch_size, mem_size)
        Key vector emitted by the controller, same shape of memory dimension.
    key_strength: Tensor (batch_size, 1)
        Key strength emitted by the controller
    interpolation_factor: Tensor (batch_size, 1)
        location based addressing step 1
    shift_weighting: Tensor (batch_size, allowed_int_shift)
        location based addressing step 2
    sharpen_factor: Tensor (batch_size, 1)
        location based addressing step 3
    weighting: Tensor (batch_size, mem_size)
        read or write weighting from previous time step
    memory: Tensor (batch_size, mem_size, mem_dim)
        memory at current time step

    Returns: w: Tensor (batch_size, mem_size)
        new_weighting after addressing
    """
    # 3.3.1
    content_based_weighting = content_addressing(key_vector, key_strength, memory)

    # 3.3.2
    w = location_addressing(content_based_weighting, interpolation_factor, shift_weighting, sharpen_factor, weighting)

    return w


def content_addressing(key_vector, key_strength, memory):
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


def location_addressing(content_based_weighting, interpolation_factor, shift_weighting,
                        sharpen_factor, weighting, ):
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
        new_weighting = interpolation_factor * content_based_weighting + (tf.constant(1.0) -
                                                                          interpolation_factor) * weighting

        # 2. shifting
        new_weighting = circular_convolution(new_weighting, shift_weighting)

        # 3. sharpening
        new_weighting = tf.pow(new_weighting, sharpen_factor)
        new_weighting = new_weighting / tf.reduce_sum(new_weighting, reduction_indices=1)

    return new_weighting


