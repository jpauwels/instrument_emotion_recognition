from .tf_functions  import *
import tensorflow as tf
import tensorflow_io as tfio


def ds_fix_length(ds, length, position='start', pad_mode='constant', pad_value='0', num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda values: tf_fix_length(values, length, position, pad_mode, pad_value), num_parallel_calls)


def ds_spectrogram(ds, fft_size, window_size, step_size, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda samples: tfio.audio.spectrogram(tf.squeeze(samples), fft_size, window_size, step_size, name='spectrogram'), num_parallel_calls)


def ds_melscale(ds, samplerate, mel_bands, fmin, fmax, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda magnitudes: tfio.audio.melscale(magnitudes, samplerate, mel_bands, fmin, fmax, name='melscale'), num_parallel_calls)


def ds_dbscale(ds, db_range=120, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda values: tfio.audio.dbscale(values, db_range, name='dbscale'), num_parallel_calls)


def ds_expand_channel(ds, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda values: tf.expand_dims(values, axis=-1), num_parallel_calls)


def ds_zscore(ds, epsilon, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda values: tf_zscore(values, epsilon), num_parallel_calls)


def ds_resample(ds, requested_rate, rate=None, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    if rate:
        return ds.map(lambda samples: tfio.audio.resample(samples, rate_in=tf.cast(rate, tf.int64), rate_out=requested_rate, name='resampler'), num_parallel_calls)
    else:
        return ds.map(lambda samples_rate: tfio.audio.resample(samples_rate[0], rate_in=tf.cast(samples_rate[1], tf.int64), rate_out=requested_rate, name='resampler'), num_parallel_calls)


def ds_melspectrogram_db(ds, src_samplerate, target_samplerate, fft_size, window_size, step_size, mel_bands, fmin=0, fmax=8000, db_range=80, epsilon=1e-10, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds\
        .apply(lambda ds: ds_resample(ds, target_samplerate, src_samplerate, num_parallel_calls))\
        .apply(lambda ds: ds_spectrogram(ds, fft_size, window_size, step_size, num_parallel_calls))\
        .apply(lambda ds: ds_melscale(ds, target_samplerate, mel_bands, fmin, fmax, num_parallel_calls))\
        .apply(lambda ds: ds_dbscale(ds, db_range, num_parallel_calls))\
#         .apply(lambda ds: ds_zscore(ds, epsilon, num_parallel_calls))\
#         .apply(lambda ds: ds_fix_length(ds, length, 'start', 'constant', '0', num_parallel_calls))\
#         .apply(lambda ds: ds_expand_channel(ds, num_parallel_calls))


def ds_slice_to_ex(ds, num_features, slice_length, start=tf.constant(-1), num_parallel_calls=tf.data.experimental.AUTOTUNE):
    def slice_steps_to_ds(tensor, label, num_features, slice_length, start):
        slices = tf_step_slicer(tensor, num_features, slice_length, start)
        slices_ds = tf.data.Dataset.from_tensor_slices(slices)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.repeat(label, len(slices)))
        return tf.data.Dataset.zip((slices_ds, labels_ds))
    return ds.map(lambda tensor, label: slice_steps_to_ds(tensor, label, num_features, slice_length, start), num_parallel_calls=num_parallel_calls)


def ds_slice_to_dim(ds, num_features, slice_length, start=tf.constant(-1), num_parallel_calls=tf.data.experimental.AUTOTUNE):
    return ds.map(lambda tensor, label: (tf_step_slicer(tensor, num_features, slice_length, start), label), num_parallel_calls=num_parallel_calls)


def ds_value_encoder(ds, key_tensor, value_tensor=None, num_parallel_calls=tf.data.experimental.AUTOTUNE):
    encoder = tf_value_encoder(key_tensor, value_tensor)
    return ds.map(lambda label: encoder.lookup(label), num_parallel_calls)


def ds_slice_dict(ds_dict, size, start=tf.constant(0)):
    return {k: v.skip(start).take(size) for k, v in ds_dict.items()}


def ds_supervised_pair(ds_dict, feature_key, label_key):
    return tf.data.Dataset.zip((ds_dict[feature_key], ds_dict[label_key]))


def ds_size(ds):
    cardinality = ds.cardinality().numpy()
    if cardinality in (tf.data.INFINITE_CARDINALITY, tf.data.UNKNOWN_CARDINALITY):
        size = 0
        for _ in ds:
            size += 1
        return size
    else:
        return cardinality