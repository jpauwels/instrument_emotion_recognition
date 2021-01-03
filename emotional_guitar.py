#!/usr/bin/env python3
from deepsuite.ds_functions import *
from deepsuite.tf_functions import *
from deepsuite.plotting import plot_confusion_matrix, mpl_fig_to_tf_image
from deepsuite.keras_functions import get_confusion_matrix
import tensorflow as tf
import tensorflow_datasets as tfds
import guitar_emotion_recognition
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from datetime import datetime
import glob
import os
import distutils
from collections import Counter


# Hyperparameter domains
hparam_domains = {}
hparam_domains['mel_bands'] = hp.HParam('mel_bands', hp.Discrete([96]))
hparam_domains['num_frames'] = hp.HParam('num_frames', hp.Discrete([187]))
hparam_domains['samplerate'] = hp.HParam('samplerate', hp.Discrete([16000]))
hparam_domains['frame_size'] = hp.HParam('frame_size', hp.Discrete([512]))
hparam_domains['step_size'] = hp.HParam('step_size', hp.Discrete([256]))
hparam_domains['feature_type'] = hp.HParam('feature_type', hp.Discrete(['essentia', 'tfds']))

hparam_domains['batch_size'] = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 256, 512]))
hparam_domains['classifier_activation'] = hp.HParam('classifier_activation', hp.Discrete(['linear', 'relu']))
hparam_domains['final_activation'] = hp.HParam('final_activation', hp.Discrete(['linear', 'softmax', 'sigmoid']))
hparam_domains['weights'] = hp.HParam('weights', hp.Discrete(['', 'MTT_musicnn_4class_transfer_learning', 'MSD_musicnn_4class_transfer_learning']))
hparam_domains['finetuning'] = hp.HParam('finetuning', hp.Discrete([True, False]))
hparam_domains['optimizer'] = hp.HParam('optimizer', hp.Discrete(['Adam', 'SGD']))
hparam_domains['learning_rate'] = hp.HParam('learning_rate', hp.Discrete([0.01, 0.001, 0.0001]))

METRIC_ACCURACY = 'sparse_categorical_accuracy'


def write_hparam_domains(log_dir):
    if not glob.glob(os.path.join(log_dir, '*.hparams.v2')):
        with tf.summary.create_file_writer(log_dir, filename_suffix='.hparams.v2').as_default():
            hp.hparams_config(
                hparams=list(hparam_domains.values()),
                metrics=[hp.Metric(METRIC_ACCURACY, group='', display_name='Train Set Accuracy', dataset_type=hp.Metric.TRAINING), 
                         hp.Metric('val_'+METRIC_ACCURACY, group='', display_name='Validation Set Accuracy', dataset_type=hp.Metric.VALIDATION),
                         hp.Metric('best_epoch', group='', display_name='Best Epoch', dataset_type=hp.Metric.VALIDATION)],
            )


def get_features(hparams):
    if hparams['feature_type'] == 'tfds':
        raw_train_ds, ds_info = tfds.load('guitar_emotion_recognition', split='train', shuffle_files=True, with_info=True, as_supervised=True)
        raw_val_ds = tfds.load('guitar_emotion_recognition', split='validation', shuffle_files=False, with_info=False, as_supervised=True)
        emotions = ds_info.features['emotion'].names

        train_ds = raw_train_ds.apply(lambda ds: ds_melspectrogram_db(ds, ds_info.features['audio'].sample_rate, hparams['samplerate'], hparams['frame_size'], hparams['frame_size'], hparams['step_size'], hparams['mel_bands']))
        val_ds = raw_val_ds.apply(lambda ds: ds_melspectrogram_db(ds, ds_info.features['audio'].sample_rate, hparams['samplerate'], hparams['frame_size'], hparams['frame_size'], hparams['step_size'], hparams['mel_bands']))
        # test_ds = raw_test_ds.apply(lambda ds: ds_melspectrogram_db(ds, ds_info.features['audio'].sample_rate, hparams['samplerate'], hparams['frame_size'], hparams['frame_size'], hparams['step_size'], hparams['mel_bands'])).apply(lambda ds: ds_time_slicer(ds, hparams['num_frames'], 0))

    elif hparams['feature_type'] == 'essentia':
         
        from essentia.streaming import MonoLoader, FrameCutter, TensorflowInputMusiCNN
        import essentia
        essentia.log.infoActive = False

        def melspectrogram_essentia(path, samplerate, frame_size, step_size):
            loader = MonoLoader(filename=path, sampleRate=float(samplerate))
            fc = FrameCutter(frameSize=int(frame_size), hopSize=int(step_size), startFromZero=True, validFrameThresholdRatio=1)
            extractor = TensorflowInputMusiCNN()
            pool = essentia.Pool()

            loader.audio >> fc.signal
            fc.frame >> extractor.frame
            extractor.bands >> (pool, "melbands")

            essentia.run(loader)
            return pool['melbands']


        def ds_filepath(ds, basedir, num_parallel_calls=tf.data.experimental.AUTOTUNE):
            def tf_filepath(basedir, filename):
                return tf.strings.join([basedir, '/emotional_guitar_dataset/', filename])
            return ds.map(lambda filename, performer, emotion: tf_filepath(basedir, filename), num_parallel_calls)


        def ds_melspectrogram_essentia(ds, samplerate, frame_size, step_size, num_parallel_calls=tf.data.experimental.AUTOTUNE):
            def tf_melspectrogram_essentia(path, samplerate, frame_size, step_size):
                melbands, = tf.py_function(tf_datatype_wrapper(melspectrogram_essentia), [path, samplerate, frame_size, step_size], [tf.float32])
                melbands.set_shape((None, hparams['mel_bands']))
                return melbands
            return ds.map(lambda path: tf_melspectrogram_essentia(path, samplerate, frame_size, step_size), num_parallel_calls)

        basedir = '/Users/johan/Datasets/Emotional guitar dataset'
        emotions = ['aggressive', 'happy', 'relaxed', 'sad']
        performers = tf.constant(['LucTur', 'DavBen', 'OweWin', 'ValFui', 'AdoLaV', 'MatRig', 'TomCan', 'TizCam', 'SteRom', 'SimArm', 'SamLor', 'AleMar', 'MasChi', 'FilMel', 'GioAcq', 'TizBol', 'SalOli', 'FraSet', 'FedCer', 'CesSam', 'AntPao', 'DavRos', 'FraBen', 'GiaFer', 'GioDic', 'NicCon', 'AntDel', 'NicLat', 'LucFra', 'AngLoi', 'MarPia'])
        csv_ds = tf.data.experimental.CsvDataset(os.path.join(basedir, 'annotations_emotional_guitar_dataset.csv'), [tf.string, tf.string, tf.string], header=True, select_cols=[1, 2, 4])
        melspec_ds = csv_ds.apply(lambda ds: ds_filepath(ds, basedir)).apply(lambda ds: ds_melspectrogram_essentia(ds, hparams['samplerate'], hparams['frame_size'], hparams['step_size']))
        emotion_ds = csv_ds.map(lambda filename, performer, emotion: emotion).apply(lambda ds: ds_value_encoder(ds, emotions))
        performer_ds = csv_ds.map(lambda filename, performer, emotion: performer).apply(lambda ds: ds_value_encoder(ds, performers))

        all_dict = {'features': melspec_ds, 'labels': emotion_ds, 'groups': performer_ds}
        train_ds_dict = ds_slice_dict(all_dict, 319)
        val_ds_dict = ds_slice_dict(all_dict, 84, 319)
        train_ds = ds_supervised_pair(train_ds_dict, 'features', 'labels').cache()
        val_ds = ds_supervised_pair(val_ds_dict, 'features', 'labels').cache()

    else:
        raise ValueError('Unknown feature type "{}"'.format(hparams['feature_type']))
        
    return emotions, train_ds, val_ds


def get_model(hparams, num_classes):
    from keras_audio_models.models import build_musicnn_classifier

    inputs = tf.keras.Input(shape=(hparams['num_frames'], hparams['mel_bands']), name='input')
    model = build_musicnn_classifier(inputs, num_classes, 100, hparams['final_activation'])
    model.get_layer('backend').logits.activation = tf.keras.activations.get(hparams['classifier_activation'])
    model.load_weights('keras_audio_models/{}.h5'.format(hparams['weights']))

    if not hparams['finetuning']:
        model.get_layer('frontend').trainable = False
        model.get_layer('midend').trainable = False
        model.get_layer('backend').bn_flat_pool.trainable = False
        model.get_layer('backend').penultimate.trainable = False
        model.get_layer('backend').bn_penultimate.trainable = False

    from_logits = model.get_layer('classifier').activation == tf.keras.activations.linear
    optimizer = tf.keras.optimizers.get({'class_name': hparams['optimizer'], 'config': {'learning_rate': hparams['learning_rate']}})
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits), metrics=[METRIC_ACCURACY])
    
    return model


def run_experiment(hparams, log_base_dir, exp_base_name):
    class_names, train_ds, val_ds = get_features(hparams)
    num_classes = len(class_names)
    model = get_model(hparams, num_classes)
    
    def ds_step_slicer(ds, num_features, slice_length, start=-1, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        return ds.map(lambda group_idx, tensor_label: (group_idx, slice_steps_to_ds(*tensor_label, num_features, slice_length, start)), num_parallel_calls=num_parallel_calls)

    def select_2nd(first, second):
        return second

    train_cardinality = train_ds.cardinality().numpy()
    if train_cardinality == tf.data.INFINITE_CARDINALITY or train_cardinality == tf.data.UNKNOWN_CARDINALITY or train_cardinality < 0:
        train_cardinality = 2500
    
    train_sliced_ds = train_ds.enumerate().apply(lambda ds: ds_step_slicer(ds, hparams['mel_bands'], hparams['num_frames'], -1))
    train_pipe = train_sliced_ds\
        .flat_map(select_2nd)\
        .shuffle(train_cardinality)\
        .batch(hparams['batch_size'])\
        .prefetch(tf.data.experimental.AUTOTUNE)
    if val_ds is not None:
        val_sliced_ds = val_ds.enumerate().apply(lambda ds: ds_step_slicer(ds, hparams['mel_bands'], hparams['num_frames'], 0)).cache()
        val_pipe = val_sliced_ds\
            .flat_map(select_2nd)\
            .batch(hparams['batch_size'])\
            .cache()\
            .prefetch(tf.data.experimental.AUTOTUNE)
    else:
        val_pipe = None

    exp_name = os.path.join(exp_base_name, model.name)
    run_name = exp_name+'-'+datetime.now().strftime("%y%m%d-%H%M%S")
    log_dir = os.path.join(log_base_dir, run_name)
    os.makedirs(os.path.dirname(exp_name), exist_ok=True)

    tensorboard = TensorBoard(log_dir=log_dir, profile_batch=2)
    hparam_writer = tf.summary.create_file_writer(log_dir, filename_suffix='.hparams.v2')
    hyper_param_logger = hp.KerasCallback(hparam_writer, hparams, trial_id=run_name)
    # exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=15, decay_rate=0.4, staircase=True)
    # pcwconst_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay([13, 28, 43, 58], [0.001, 0.0004, 0.00016, 6.4e-5, 4e-5])
    # lrate = LearningRateScheduler(pcwconst_decay, verbose=1)
    # from deepsuite.keras import ConfusionMatrixOnEpoch
    # confusion_matrix = ConfusionMatrixOnEpoch(log_dir, class_names, ['train', 'validation'])
    callbacks = [tensorboard, hyper_param_logger]#[, confusion_matrix, lrate],
    if val_ds is not None:
        earlystopper = EarlyStopping(monitor='val_'+METRIC_ACCURACY, patience=hparams['early_stopping_patience'], verbose=1, restore_best_weights=True)
        checkpointer = ModelCheckpoint(exp_name+'.h5', monitor='val_'+METRIC_ACCURACY, verbose=1, save_best_only=True)
        callbacks += [earlystopper, checkpointer]

    results = model.fit(train_pipe, validation_data=val_pipe, epochs=hparams['epochs'], verbose=2, callbacks=callbacks)
    if os.path.exists(exp_name+'.h5'):
        model.load_weights(exp_name+'.h5') # when max epochs gets exceeded, early stopping callback doesn't load best model

    def majority_voting(sliced_ds):
        soft_metrics = []
        hard_metrics = []
        soft_voting_labels = []
        hard_voting_labels = []
        # true_labels = tf.concat((*nonsliced_ds.map(lambda _, label: label),), axis=0)
        true_labels = []
        metric = tf.keras.metrics.get(METRIC_ACCURACY)
        for _, file_slices in sliced_ds:
            true_label = next(iter(file_slices))[1].numpy()
            file_results = model.predict(file_slices.batch(hparams['batch_size']), verbose=0)
            soft_results = file_results.mean(axis=0)
            hard_metrics.append(tf.reduce_mean(metric(y_true=tf.repeat(true_label, file_results.shape[0]), y_pred=file_results)))
            soft_metrics.append(metric(y_true=true_label, y_pred=soft_results))
            true_labels.append(true_label)
            soft_voting_labels.append(soft_results.argmax())
            hard_voting_labels.append(next(Counter(file_results.argmax(axis=1)).elements()))
        hard_metric = tf.reduce_mean(hard_metrics).numpy()
        soft_metric = tf.reduce_mean(soft_metrics).numpy()
        return soft_metric, hard_metric, true_labels, soft_voting_labels, hard_voting_labels

    # Write best model summaries
    with tf.summary.create_file_writer(log_dir, filename_suffix='.final.v2').as_default():
        best_epoch = results.history[earlystopper.monitor].index(earlystopper.best)
        tf.summary.scalar('best_epoch', best_epoch, step=0)
        results_train = model.evaluate(train_pipe, verbose=0)
        results_val = model.evaluate(val_pipe, verbose=0)
        for name, train, val in zip(model.metrics_names, results_train, results_val):
            tf.summary.scalar(name, train, step=0)
            tf.summary.scalar('val_'+name, val, step=0)
            print('The final model has achieved a {} of {:.3f} and a {} of {:.3f} at epoch {}'.format(name, train, 'val_'+name, val, best_epoch+1))
        train_conf = get_confusion_matrix(model, train_pipe, class_names, normalize=True, title='')
        tf.summary.image('Train Set Confusion', mpl_fig_to_tf_image(train_conf), step=best_epoch)
        val_conf = get_confusion_matrix(model, val_pipe, class_names, normalize=True, title='')
        tf.summary.image('Validation Set Confusion', mpl_fig_to_tf_image(val_conf), step=best_epoch)

        metric_name = tf.keras.metrics.get(METRIC_ACCURACY).__name__.replace('_', ' ')
        for sliced_ds, ds_name in [(train_sliced_ds, 'train'), (val_sliced_ds, 'validation')]:
            soft_metric, hard_metric, true_labels, soft_voting_labels, hard_voting_labels = majority_voting(sliced_ds)
            tf.summary.scalar(f'{ds_name.title()} Soft Voting {metric_name.title()}', soft_metric, step=0)
            tf.summary.scalar(f'{ds_name.title()} Hard Voting {metric_name.title()}', hard_metric, step=0)
            print(f'The final model has achieved a {ds_name} soft voting {metric_name} of {soft_metric:.3f} and a {ds_name} hard voting {metric_name} of {hard_metric:.3f} at epoch {best_epoch+1}')
            soft_conf = plot_confusion_matrix(tf.math.confusion_matrix(true_labels, soft_voting_labels), class_names, normalize=True)
            hard_conf = plot_confusion_matrix(tf.math.confusion_matrix(true_labels, hard_voting_labels), class_names, normalize=True)
            tf.summary.image(f'{ds_name} Soft Voting Confusion', mpl_fig_to_tf_image(soft_conf), step=best_epoch)
            tf.summary.image(f'{ds_name} Hard Voting Confusion', mpl_fig_to_tf_image(hard_conf), step=best_epoch)
        
        
if __name__ == '__main__':
    
    import argparse
    import itertools
    
    log_base_dir = os.path.expanduser('~/private/tensorboard')
    exp_base_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join(log_base_dir, exp_base_name)

    class ResetAppendAction(argparse._AppendAction):
        def __init__(self, option_strings, dest, **kwargs):
            self.reset = True
            super(ResetAppendAction, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if self.reset:
                setattr(namespace, self.dest, [])
                self.reset = False
            super(ResetAppendAction, self).__call__(parser, namespace, values, option_string)

    parser = argparse.ArgumentParser(description='''
    Run emotional guitar classification experiment.''',
    allow_abbrev=False)
    features_config = parser.add_argument_group('Feature options')
    features_config.add_argument('-m', '--mel-bands', default=[96], action=ResetAppendAction, type=int)
    features_config.add_argument('-n', '--num-frames', default=[187], action=ResetAppendAction, type=int)
    features_config.add_argument('-r', '--samplerate', default=[16000], action=ResetAppendAction, type=int)
    features_config.add_argument('-f', '--frame-size', default=[512], action=ResetAppendAction, type=int)
    features_config.add_argument('-s', '--step-size', default=[256], action=ResetAppendAction, type=int)
    features_config.add_argument('-t', '--feature-type', default=['essentia'], action=ResetAppendAction, type=str, choices=hparam_domains['feature_type'].domain.values)
    
    model_config = parser.add_argument_group('Model options')
    model_config.add_argument('-c', '--classifier-activation', default=['relu'], action=ResetAppendAction, type=str, choices=hparam_domains['classifier_activation'].domain.values)
    model_config.add_argument('-a', '--final-activation', default=['softmax'], action=ResetAppendAction, type=str, choices=hparam_domains['final_activation'].domain.values)
    model_config.add_argument('-w', '--weights', default=['MTT_musicnn_4class_transfer_learning'], action=ResetAppendAction, type=str, choices=hparam_domains['weights'].domain.values)
    model_config.add_argument('--finetuning', default=[False], action=ResetAppendAction, type=lambda x: bool(distutils.util.strtobool(x)))
    model_config.add_argument('-o', '--optimizer', default=['Adam'], action=ResetAppendAction, type=str, choices=hparam_domains['optimizer'].domain.values)
    model_config.add_argument('-l', '--learning-rate', default=[0.001], action=ResetAppendAction, type=float)
    model_config.add_argument('-b', '--batch-size', default=[256], action=ResetAppendAction, type=int)
    
    train_config = parser.add_argument_group('Training options')
#     train_config.add_argument('--validation_split', default=[0.2], action=ResetAppendAction, type=float)
    train_config.add_argument('-e', '--epochs', default=[100], action=ResetAppendAction, type=int)
    train_config.add_argument('--early_stopping_patience', default=[30], action=ResetAppendAction, type=int)

    args = vars(parser.parse_args())
    write_hparam_domains(log_dir)

    for param_combo in itertools.product(*args.values()):
        hparams = dict(zip(args.keys(), param_combo))
        print(f'Running parameter combination {", ".join(["{}: {}".format(k, v) for k, v in hparams.items()])}')
        run_experiment(hparams, log_base_dir, exp_base_name)
