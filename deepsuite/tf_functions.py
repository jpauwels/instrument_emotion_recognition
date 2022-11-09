import tensorflow as tf


def tf_datatype_wrapper(func):
    def tf_convert_type(tf_type):
        try:
            np_type = tf_type.numpy()
        except AttributeError:
            return tf_type
        try:
            return np_type.decode()
        except AttributeError:
            return np_type

    def wrap_and_call(*args, **kwargs): 
        new_args = []
        for arg in args:
            new_args.append(tf_convert_type(arg))
        new_kwargs = {}
        for k, v in kwargs.items():
            new_kwargs[k] = tf_convert_type(v)
        return func(*new_args, **new_kwargs)

    return wrap_and_call


def tf_fix_length(tensor, length, position='start', pad_mode='constant', pad_value='0'):
    len_diff = tf.shape(tensor)[0] - length
    if len_diff > 0:
        if position == 'start':
            return tensor[:-len_diff]
        elif position == 'middle':
            return tensor[len_diff//2:-len_diff//2]
        else:
            return tensor[len_diff:]
    elif len_diff < 0:
        non_padded_dims = tf.tile([(0, 0)], [tf.rank(tensor)-1, 1])
        if position == 'start':
            paddings = tf.concat(([(0, -len_diff)], non_padded_dims), axis=0)
        elif position == 'middle':
            paddings = tf.concat(([(tf.math.ceil(-len_diff/2), -len_diff//2)], non_padded_dims), axis=0)
        else:
            paddings = tf.concat(([(-len_diff, 0)], non_padded_dims), axis=0)
        if pad_value == 'min':
            pad_value = tf.reduce_min(tensor)
        elif pad_value == 'max':
            pad_value = tf.reduce_max(tensor)
        elif pad_value == 'mean':
            pad_value = tf.reduce_mean(tensor)
        else:
            pad_value = tf.strings.to_number(pad_value)
        return tf.pad(tensor, paddings, mode=pad_mode, constant_values=pad_value, name='padding')
    else:
        return tensor


def tf_zscore(tensor, epsilon):
    eps = tf.constant(epsilon)
    return (tensor - tf.math.reduce_mean(tensor, keepdims=True)) / tf.math.maximum(tf.math.reduce_std(tensor, keepdims=True), eps)


def tf_step_slicer(tensor, label, num_features, slice_length, start=tf.constant(-1)):
    @tf_datatype_wrapper
    def step_slicer(tensor, label, slice_length, start):
        num_steps, num_features = tf.shape(tensor)
        num_slices = num_steps // slice_length
        if start < 0:
            used_steps = num_slices * slice_length
            tensor = tf.image.random_crop(tensor, [used_steps, num_features])
        elif start > 0:
            tensor = tensor[start:]
        tensor_expanded = tf.expand_dims(tf.expand_dims(tensor, axis=-1), axis=0)
        flat_slices = tf.image.extract_patches(tensor_expanded, sizes=[1, slice_length, num_features, 1], strides=[1, slice_length, num_features, 1], rates=[1, 1, 1, 1], padding='VALID', name='time_slicer')
        slices = tf.reshape(flat_slices, [num_slices, slice_length, num_features])
        return slices, tf.repeat(label, num_slices)

    slices, labels = tf.py_function(step_slicer, [tensor, label, slice_length, start], [tf.float32, tf.int32])
    slices.set_shape((None, slice_length, num_features))
    labels.set_shape((None,))
    return slices, labels


def tf_value_encoder(key_tensor, value_tensor=None):
    if value_tensor is None:
        value_tensor = tf.range(key_tensor.shape[0])
    return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(key_tensor, value_tensor), default_value=-1)
