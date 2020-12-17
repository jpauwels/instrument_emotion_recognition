from .tf_functions  import *
import tensorflow as tf
import tensorflow_io as tfio


def ds_fix_length(ds, length, position='start', pad_mode='constant', pad_value='0', num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda audio, label: (tf_fix_length(audio, length, position, pad_mode, pad_value), label), num_parallel_calls)


def ds_spectrogram(ds, fft_size, window_size, step_size, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda samples, label: (tfio.experimental.audio.spectrogram(tf.squeeze(samples), fft_size, window_size, step_size, name='spectrogram'), label), num_parallel_calls)


def ds_melscale(ds, samplerate, mel_bands, fmin, fmax, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda magnitudes, label: (tfio.experimental.audio.melscale(magnitudes, samplerate, mel_bands, fmin, fmax, name='melscale'), label), num_parallel_calls)


def ds_dbscale(ds, db_range=120, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda values, label: (tfio.experimental.audio.dbscale(values, db_range, name='dbscale'), label), num_parallel_calls)


def ds_expand_channel(ds, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda values, label: (tf.expand_dims(values, axis=-1), label), num_parallel_calls)


def ds_zscore(ds, epsilon, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda values, label: (tf_zscore(values, epsilon), label), num_parallel_calls)


def ds_resample(ds, requested_rate, rate=None, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    if rate:
        return ds.map(lambda samples, label: (tfio.audio.resample(samples, rate_in=tf.cast(rate, tf.int64), rate_out=requested_rate, name='resampler'), label), num_parallel_calls)
    else:
        return ds.map(lambda samples_rate, label: (tfio.audio.resample(samples_rate[0], rate_in=tf.cast(samples_rate[1], tf.int64), rate_out=requested_rate, name='resampler'), label), num_parallel_calls)

    
def ds_melspectrogram_db(ds, src_samplerate, target_samplerate, fft_size, window_size, step_size, mel_bands, fmin=0, fmax=8000, db_range=80, epsilon=1e-10, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds\
        .apply(lambda ds: ds_resample(ds, target_samplerate, src_samplerate, num_parallel_calls))\
        .apply(lambda ds: ds_spectrogram(ds, fft_size, window_size, step_size, num_parallel_calls))\
        .apply(lambda ds: ds_melscale(ds, target_samplerate, mel_bands, fmin, fmax, num_parallel_calls))\
        .apply(lambda ds: ds_dbscale(ds, db_range, num_parallel_calls))\
#         .apply(lambda ds: ds_zscore(ds, epsilon, num_parallel_calls))\
#         .apply(lambda ds: ds_fix_length(ds, length, 'start', 'constant', '0', num_parallel_calls))\
#         .apply(lambda ds: ds_expand_channel(ds, num_parallel_calls))


def ds_time_slicer(ds, slice_length, start=-1, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    def variable_time_slicer(tensor, label):
        slices, labels = tf.py_function(tf_time_slicer, [tensor, label, slice_length, start], [tf.float32, tf.int64])
        slices.set_shape((None, None, 96))
        labels.set_shape((None,))
        slices_ds = tf.data.Dataset.from_tensor_slices(slices)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((slices_ds, labels_ds))
    return ds.flat_map(variable_time_slicer)


def ds_encode_binary(ds, category):
    def tf_encode_binary(label, category):
        if label == category:
            return tf.constant(0, dtype=tf.int64)
        else:
            return tf.constant(1, dtype=tf.int64)
    return ds.map(lambda tensor, label: (tensor, tf_encode_binary(label, category)))
