#!/usr/bin/env python3
from essentia.streaming import VectorInput, FrameCutter, TensorflowInputMusiCNN
import essentia
essentia.log.infoActive = False
from deepsuite.ds_functions import slice_steps_to_ds
from deepsuite.tf_functions import tf_datatype_wrapper
from deepsuite.plotting import plot_confusion_matrix, mpl_fig_to_tf_image
from deepsuite.keras_functions import get_pred_labels
import tensorflow as tf
import tensorflow_datasets as tfds
import guitar_emotion_recognition
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import numpy as np
from datetime import datetime
import glob
import os
from collections import Counter
import tempfile
import csv


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
hparam_domains['weights'] = hp.HParam('weights', hp.Discrete(['', 'MTT_musicnn', 'MSD_musicnn']))
hparam_domains['finetuning'] = hp.HParam('finetuning', hp.Discrete([True, False]))
hparam_domains['optimizer'] = hp.HParam('optimizer', hp.Discrete(['Adam', 'SGD']))
hparam_domains['learning_rate'] = hp.HParam('learning_rate', hp.Discrete([0.01, 0.001, 0.0001]))

METRIC_ACCURACY = 'sparse_categorical_accuracy'
accurary_name = tf.keras.metrics.get(METRIC_ACCURACY).__name__.replace('_', ' ')


def write_hparam_domains(log_dir):
    for p in glob.glob(os.path.join(log_dir, '*.hparam_domains.v2')):
        os.remove(p)
    hparam_metrics = []
    for split_name, split_type in (('validation', hp.Metric.VALIDATION), ('train', hp.Metric.TRAINING)):
        hparam_metrics.append(hp.Metric(f'{split_name}.{METRIC_ACCURACY}', group='', display_name=f'{split_name.title()} {accurary_name.title()}', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.{METRIC_ACCURACY}_std', group='', display_name=f'{split_name.title()} {accurary_name.title()} StDev', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.soft_voting_{METRIC_ACCURACY}', group='', display_name=f'{split_name.title()} Soft Voting {accurary_name.title()}', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.soft_voting_{METRIC_ACCURACY}_std', group='', display_name=f'{split_name.title()} Soft Voting {accurary_name.title()} StDev', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.hard_voting_{METRIC_ACCURACY}', group='', display_name=f'{split_name.title()} Hard Voting {accurary_name.title()}', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.hard_voting_{METRIC_ACCURACY}_std', group='', display_name=f'{split_name.title()} Hard Voting {accurary_name.title()} StDev', dataset_type=split_type))
    hparam_metrics.append(hp.Metric('best_epoch', group='', display_name='Best Epoch', dataset_type=hp.Metric.VALIDATION))
    with tf.summary.create_file_writer(log_dir, filename_suffix='.hparam_domains.v2').as_default():
        hp.hparams_config(hparams=list(hparam_domains.values()), metrics=hparam_metrics)


def get_features(hparams, paths, labels, train_indices, val_indices):
    if hparams['feature_type'] == 'tfds':
        train_ds = tfds.load('guitar_emotion_recognition', split='train', shuffle_files=True, with_info=False, as_supervised=True)
        val_ds = tfds.load('guitar_emotion_recognition', split='validation', shuffle_files=False, with_info=False, as_supervised=True)

    elif hparams['feature_type'] == 'essentia':
        from essentia.standard import MonoLoader

        def ds_audioloader_essentia(ds, samplerate, num_parallel_calls=tf.data.experimental.AUTOTUNE):
            def audioloader_essentia(path, samplerate):
                return MonoLoader(filename=path, sampleRate=float(samplerate))()

            def tf_audioloader_essentia(path, samplerate):
                audio, = tf.py_function(tf_datatype_wrapper(audioloader_essentia), [path, samplerate], [tf.float32])
                audio.set_shape((None))
                return audio

            return ds.map(lambda path: tf_audioloader_essentia(path, samplerate), num_parallel_calls)

        train_features_ds = tf.data.Dataset.from_tensor_slices(paths[train_indices]).apply(lambda ds: ds_audioloader_essentia(ds, hparams['samplerate']))
        val_features_ds = tf.data.Dataset.from_tensor_slices(paths[val_indices]).apply(lambda ds: ds_audioloader_essentia(ds, hparams['samplerate']))
        train_labels_ds = tf.data.Dataset.from_tensor_slices(labels[train_indices])
        val_labels_ds = tf.data.Dataset.from_tensor_slices(labels[val_indices])
        train_ds = tf.data.Dataset.zip((train_features_ds, train_labels_ds))
        val_ds = tf.data.Dataset.zip((val_features_ds, val_labels_ds))
        
        def ds_melspectrogram_essentia(ds, frame_size, step_size, num_parallel_calls=tf.data.experimental.AUTOTUNE):
            def melspectrogram_essentia(audio, frame_size, step_size):
                audio_input = VectorInput(audio)
                fc = FrameCutter(frameSize=int(frame_size), hopSize=int(step_size), startFromZero=True, validFrameThresholdRatio=1)
                extractor = TensorflowInputMusiCNN()
                pool = essentia.Pool()

                audio_input.data >> fc.signal
                fc.frame >> extractor.frame
                extractor.bands >> (pool, "melbands")

                essentia.run(audio_input)
                return pool['melbands']

            def tf_melspectrogram_essentia(audio, frame_size, step_size):
                melbands, = tf.py_function(tf_datatype_wrapper(melspectrogram_essentia), [audio, frame_size, step_size], [tf.float32])
                melbands.set_shape((None, hparams['mel_bands']))
                return melbands
            return ds.map(lambda audio, label: (tf_melspectrogram_essentia(audio, frame_size, step_size), label), num_parallel_calls)

        train_ds = train_ds.apply(lambda ds: ds_melspectrogram_essentia(ds, hparams['frame_size'], hparams['step_size']))
        val_ds = val_ds.apply(lambda ds: ds_melspectrogram_essentia(ds, hparams['frame_size'], hparams['step_size']))
    else:
        raise ValueError('Unknown feature type "{}"'.format(hparams['feature_type']))

    return train_ds, val_ds


def get_model(hparams, num_classes):
    from keras_audio_models.models import build_musicnn_classifier

    inputs = tf.keras.Input(shape=(hparams['num_frames'], hparams['mel_bands']), name='input')
    model = build_musicnn_classifier(inputs, num_classes, 100, hparams['classifier_activation'], hparams['final_activation'], weights=hparams['weights'])

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


def majority_voting(model, sliced_ds):
    soft_metrics = []
    hard_metrics = []
    soft_voting_labels = []
    hard_voting_labels = []
    # true_labels = tf.concat((*nonsliced_ds.map(lambda _, label: label),), axis=0)
    true_labels = []
    metric = tf.keras.metrics.get(METRIC_ACCURACY)
    for file_slices in sliced_ds:
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


def fit_model(model, exp_name, log_dir, save_model_dir, train_ds, val_ds, log_suffix=''):

    def ds_step_slicer(ds, num_features, slice_length, start=-1, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        return ds.map(lambda tensor, label: slice_steps_to_ds(tensor, label, num_features, slice_length, start), num_parallel_calls=num_parallel_calls)

    sliced_ds = {'train': train_ds.cache().apply(lambda ds: ds_step_slicer(ds, hparams['mel_bands'], hparams['num_frames'], -1))}

    train_cardinality = sliced_ds['train'].flat_map(lambda x: x).cardinality().numpy()
    if train_cardinality == tf.data.INFINITE_CARDINALITY or train_cardinality == tf.data.UNKNOWN_CARDINALITY or train_cardinality < 0:
        train_cardinality = 3000
    
    train_pipe = sliced_ds['train']\
        .flat_map(lambda x: x)\
        .shuffle(train_cardinality)\
        .batch(hparams['batch_size'])\
        .prefetch(tf.data.experimental.AUTOTUNE)
    if val_ds is not None:
        sliced_ds['validation'] = val_ds.apply(lambda ds: ds_step_slicer(ds, hparams['mel_bands'], hparams['num_frames'], 0)).cache()
        val_pipe = sliced_ds['validation']\
            .flat_map(lambda x: x)\
            .batch(hparams['batch_size'])\
            .cache()\
            .prefetch(tf.data.experimental.AUTOTUNE)
    else:
        val_pipe = None

    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    tensorboard = TensorBoard(log_dir=log_dir+log_suffix, profile_batch=0)
    # exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=15, decay_rate=0.4, staircase=True)
    # pcwconst_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay([13, 28, 43, 58], [0.001, 0.0004, 0.00016, 6.4e-5, 4e-5])
    # lrate = LearningRateScheduler(pcwconst_decay, verbose=1)
    # from deepsuite.keras import ConfusionMatrixOnEpoch
    # confusion_matrix = ConfusionMatrixOnEpoch(log_dir, class_names, ['train', 'validation'])
    callbacks = [tensorboard]#[, confusion_matrix, lrate],
    tmp_weights_path = os.path.join(tempfile.gettempdir(), exp_name+log_suffix+'.h5')
    if val_ds is not None:
        earlystopper = EarlyStopping(monitor='val_'+METRIC_ACCURACY, patience=hparams['early_stopping_patience'], verbose=1, restore_best_weights=True)
        checkpointer = ModelCheckpoint(tmp_weights_path, monitor='val_'+METRIC_ACCURACY, verbose=1, save_best_only=True)
        callbacks += [earlystopper, checkpointer]

    fit_log = model.fit(train_pipe, validation_data=val_pipe, epochs=hparams['epochs'], verbose=2, callbacks=callbacks)
    if os.path.exists(tmp_weights_path):
        model.load_weights(tmp_weights_path) # when max epochs gets exceeded, early stopping callback doesn't load best model
        os.remove(tmp_weights_path)
    if save_model_dir:
        save_path = os.path.join(save_model_dir, exp_name+log_suffix+'.h5')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path, save_format='h5')

    eval_results = {'train': model.evaluate(train_pipe, verbose=0)}
    conf_mat = {'train': tf.math.confusion_matrix(*get_pred_labels(model, train_pipe))}
    if val_ds is not None:
        best_epoch = fit_log.history[earlystopper.monitor].index(earlystopper.best)
        eval_results['validation'] = model.evaluate(val_pipe, verbose=0)
        conf_mat['validation'] = tf.math.confusion_matrix(*get_pred_labels(model, val_pipe))
    else:
        best_epoch = hparams['epochs']

    majority_eval_results = {}
    majority_conf_mat = {}
    for name, ds in sliced_ds.items():
        soft_metric, hard_metric, true_labels, soft_voting_labels, hard_voting_labels = majority_voting(model, ds)
        majority_eval_results[name] = soft_metric, hard_metric
        soft_conf = tf.math.confusion_matrix(true_labels, soft_voting_labels)
        hard_conf = tf.math.confusion_matrix(true_labels, hard_voting_labels)
        majority_conf_mat[name] = soft_conf, hard_conf

    return best_epoch, eval_results, conf_mat, majority_eval_results, majority_conf_mat


def write_log(log_dir, hparams, exp_name, class_names, metrics_names, best_epoch, eval_results, conf_mat, majority_eval_results, majority_conf_mat, eval_stdev=None, majority_stdev=None):
    with tf.summary.create_file_writer(log_dir, filename_suffix='.final.v2').as_default():
        hp.hparams(hparams, trial_id=exp_name)
        if best_epoch is not None:
            tf.summary.scalar('best_epoch', best_epoch, step=0)
        else:
            best_epoch = 0

        for split_name, results in eval_results.items():
            for metric_name, value in zip(metrics_names, results):
                try:
                    tf.summary.scalar(f'{split_name}.{metric_name}', value[0], step=0)
                    tf.summary.scalar(f'{split_name}.{metric_name}_std', value[1], step=0)
                    print('The final model has achieved a {} of {:.3f} +/- {:.3f}'.format(metric_name, *value))
                except TypeError:
                    tf.summary.scalar(f'{split_name}.{metric_name}', value, step=0)
                    print('The final model has achieved a {} of {:.3f} at epoch {}'.format(metric_name, value, best_epoch+1))

        for split_name, conf in conf_mat.items():
            tf.summary.image(f'{split_name.title()} Confusion', mpl_fig_to_tf_image(plot_confusion_matrix(conf, class_names, normalize=True, title='')), step=best_epoch)

        for split_name, (soft_metric, hard_metric) in majority_eval_results.items():
            try:
                tf.summary.scalar(f'{split_name}.soft_voting_{METRIC_ACCURACY}', soft_metric[0], step=0)
                tf.summary.scalar(f'{split_name}.soft_voting_{METRIC_ACCURACY}_std', soft_metric[1], step=0)
                tf.summary.scalar(f'{split_name}.hard_voting_{METRIC_ACCURACY}', hard_metric[0], step=0)
                tf.summary.scalar(f'{split_name}.hard_voting_{METRIC_ACCURACY}_std', hard_metric[1], step=0)
                print(f'The final model has achieved a {split_name} soft voting {accurary_name} of {soft_metric[0]:.3f} +/- {soft_metric[1]:.3f} and a {split_name} hard voting {accurary_name} of {hard_metric[0]:.3f} +/- {hard_metric[1]:.3f}')
            except IndexError:
                tf.summary.scalar(f'{split_name}.soft_voting_{METRIC_ACCURACY}', soft_metric, step=0)
                tf.summary.scalar(f'{split_name}.hard_voting_{METRIC_ACCURACY}', hard_metric, step=0)
                print(f'The final model has achieved a {split_name} soft voting {accurary_name} of {soft_metric:.3f} and a {split_name} hard voting {accurary_name} of {hard_metric:.3f} at epoch {best_epoch+1}')
        
        for split_name, (soft_conf, hard_conf) in majority_conf_mat.items():
            tf.summary.image(f'{split_name.title()} Soft Voting Confusion', mpl_fig_to_tf_image(plot_confusion_matrix(soft_conf, class_names, normalize=True, title='')), step=best_epoch)
            tf.summary.image(f'{split_name.title()} Hard Voting Confusion', mpl_fig_to_tf_image(plot_confusion_matrix(hard_conf, class_names, normalize=True, title='')), step=best_epoch)


# def clone_model_weights(model):
#     from copy import deepcopy
#     model_copy = tf.keras.models.clone_model(model)
#     if not model.get_layer('frontend').trainable:
#         model_copy.get_layer('backend').bn_flat_pool.trainable = False
#         model_copy.get_layer('backend').penultimate.trainable = False
#         model_copy.get_layer('backend').bn_penultimate.trainable = False
#     model_copy.compile(optimizer=deepcopy(model.optimizer), loss=deepcopy(model.loss), metrics=deepcopy(model.compiled_metrics._metrics))
#     model_copy.set_weights(model.get_weights())
#     return model_copy


def run_experiment(hparams, log_base_dir, exp_base_name, save_model_dir):
    tf.random.set_seed(hparams['tf_seed'])

    basedir = os.path.join(os.getenv('TFDS_DATA_DIR', os.path.expanduser('~/tensorflow_datasets')), 'downloads/extracted/ZIP.acoustic_guitar_emotion_dataset-v0.4.0.zip')
    class_names = ['aggressive', 'relaxed', 'happy', 'sad']
    performers = ['LucTur', 'DavBen', 'OweWin', 'ValFui', 'AdoLaV', 'MatRig', 'TomCan', 'TizCam', 'SteRom', 'SimArm', 'SamLor', 'AleMar', 'MasChi', 'FilMel', 'GioAcq', 'TizBol', 'SalOli', 'FraSet', 'FedCer', 'CesSam', 'AntPao', 'DavRos', 'FraBen', 'GiaFer', 'GioDic', 'NicCon', 'AntDel', 'NicLat', 'LucFra', 'AngLoi', 'MarPia']
    with open(os.path.join(basedir, 'emotional_guitar_dataset/annotations_emotional_guitar_dataset.csv')) as f:
        rows = [row for row in csv.DictReader(f)]
    paths = np.array([os.path.join(basedir, 'emotional_guitar_dataset', row['file_name']) for row in rows])
    labels = np.array([class_names.index(row['emotion']) for row in rows], dtype=np.int32)
    groups = np.array([performers.index(row['composer_pseudonym']) for row in rows], dtype=np.int32)

    exp_name = os.path.join(exp_base_name, datetime.now().strftime("%y%m%d-%H%M%S"))
    log_dir = os.path.join(log_base_dir, exp_name)

    if hparams['validation_split']:
        train_indices, val_indices = next(GroupShuffleSplit(n_splits=1, test_size=hparams['validation_split'], random_state=hparams['split_seed']).split(paths, labels, groups))
        train_ds, val_ds = get_features(hparams, paths, labels, train_indices, val_indices)
        model = get_model(hparams, len(class_names))
        model_output = fit_model(model, exp_name, log_dir, save_model_dir, train_ds, val_ds)
        write_log(log_dir, hparams, exp_name, class_names, model.metrics_names, *model_output)

    elif hparams['num_folds']:
        fold_outputs = []
        fold_splitter = GroupKFold(n_splits=hparams['num_folds'])
        if not hparams['finetuning']:
            init_model = get_model(hparams, len(class_names))
        for train_indices, val_indices in fold_splitter.split(paths, labels, groups):
            train_ds, val_ds = get_features(hparams, paths, labels, train_indices, val_indices)
            fold_suffix = f'/fold{len(fold_outputs)+1}'
            if hparams['finetuning']:
                fold_model = get_model({**hparams, 'weights': hparams['weights']+fold_suffix}, len(class_names))
            else:
                fold_model = get_model(hparams, len(class_names))
                fold_model.set_weights(init_model.get_weights())
            fold_outputs.append(fit_model(fold_model, exp_name, log_dir, save_model_dir, train_ds, val_ds, log_suffix=fold_suffix))
        _, fold_results, fold_conf_mats, fold_majority_eval_results, fold_majority_conf_mat = zip(*fold_outputs)
        mean_eval_results = {split: list(zip(np.mean(np.stack([f[split] for f in fold_results]), axis=0), 
                                             np.std(np.stack([f[split] for f in fold_results]), axis=0))) for split in fold_results[0].keys()}
        sum_conf_mat = {split: tf.reduce_sum([f[split] for f in fold_conf_mats], axis=0) for split in fold_conf_mats[0].keys()}
        mean_majority_eval_results = {split: list(zip(np.mean(np.stack([f[split] for f in fold_majority_eval_results]), axis=0),
                                                      np.std(np.stack([f[split] for f in fold_majority_eval_results]), axis=0))) for split in fold_majority_eval_results[0].keys()}
        sum_majority_conf_mat = {split: [v for v in tf.reduce_sum([f[split] for f in fold_majority_conf_mat], axis=0)] for split in fold_majority_conf_mat[0].keys()}
        write_log(log_dir, hparams, exp_name, class_names, fold_model.metrics_names, None, mean_eval_results, sum_conf_mat, mean_majority_eval_results, sum_majority_conf_mat)



if __name__ == '__main__':
    
    import argparse
    import itertools
    from distutils.util import strtobool
    
    log_base_dir = os.path.expanduser('~/private/tensorboard')
    exp_base_name = os.path.splitext(os.path.basename(__file__))[0]

    def list_saved_models(save_model_dir, exp_base_name):
        d = os.path.join(save_model_dir, exp_base_name)
        try:
            return [os.path.join('..', d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) or os.path.splitext(x)[1] == '.h5']
        except FileNotFoundError:
            return []

    hparam_domains['weights'] = hp.HParam('weights', hp.Discrete(hparam_domains['weights'].domain.values + list_saved_models('saved-models', exp_base_name)))

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
    model_config.add_argument('-w', '--weights', default=['MTT_musicnn'], action=ResetAppendAction, type=str, choices=hparam_domains['weights'].domain.values)

    train_config = parser.add_argument_group('Training options')
    train_config.add_argument('--validation-split', default=[0], action=ResetAppendAction, type=float)
    train_config.add_argument('--num-folds', default=[0], action=ResetAppendAction, type=int)
    train_config.add_argument('--split-seed', default=[0], action=ResetAppendAction, type=int)
    train_config.add_argument('--tf-seed', default=[0], action=ResetAppendAction, type=int)
    train_config.add_argument('-o', '--optimizer', default=['Adam'], action=ResetAppendAction, type=str, choices=hparam_domains['optimizer'].domain.values)
    train_config.add_argument('-l', '--learning-rate', default=[0.001], action=ResetAppendAction, type=float)
    train_config.add_argument('-b', '--batch-size', default=[256], action=ResetAppendAction, type=int)
    train_config.add_argument('-e', '--epochs', default=[100], action=ResetAppendAction, type=int)
    train_config.add_argument('--early-stopping-patience', default=[30], action=ResetAppendAction, type=int)
    train_config.add_argument('--save-model-dir', default='', action='store', type=str)
    train_config.add_argument('--finetuning', default=[False], action=ResetAppendAction, type=lambda x: bool(strtobool(x)))

    args = vars(parser.parse_args())
    save_model_dir = args.pop('save_model_dir')
    write_hparam_domains(os.path.join(log_base_dir, exp_base_name))

    for param_combo in itertools.product(*args.values()):
        hparams = dict(zip(args.keys(), param_combo))
        print(f'Running parameter combination {", ".join(["{}: {}".format(k, v) for k, v in hparams.items()])}')
        run_experiment(hparams, log_base_dir, exp_base_name, save_model_dir)
