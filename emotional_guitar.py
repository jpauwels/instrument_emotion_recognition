#!/usr/bin/env python3
from essentia.streaming import VectorInput, FrameCutter, TensorflowInputMusiCNN
import essentia
essentia.log.infoActive = False
from deepsuite.ds_functions import ds_step_slicer
from deepsuite.tf_functions import tf_datatype_wrapper
from deepsuite.plotting import plot_confusion_matrix, mpl_fig_to_tf_image
from deepsuite.keras_functions import get_pred_labels
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
from instrument_emotion_datasets import acoustic_guitar_emotion_recognition, electric_guitar_emotion_recognition, piano_emotion_recognition
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
import numpy as np
from datetime import datetime
from collections import Counter
import tempfile

tfdata_parallel = tf.data.experimental.AUTOTUNE

# Hyperparameter domains
hparam_domains = {}
hparam_domains['instrument'] = hp.HParam('instrument', hp.Discrete(['acoustic_guitar', 'electric_guitar', 'piano']))
hparam_domains['weights'] = hp.HParam('weights', hp.Discrete(['', 'MTT_musicnn', 'MSD_musicnn']))
hparam_domains['finetuning'] = hp.HParam('finetuning', hp.Discrete([True, False]))
hparam_domains['learning_rate'] = hp.HParam('learning_rate', hp.Discrete([0.01, 0.001, 0.0001, 0.00001, 0.000001]))
hparam_domains['batch_size'] = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 256, 512]))

hparam_domains['classifier_activation'] = hp.HParam('classifier_activation', hp.Discrete(['linear', 'relu']))
hparam_domains['final_activation'] = hp.HParam('final_activation', hp.Discrete(['linear', 'softmax', 'sigmoid']))
hparam_domains['optimizer'] = hp.HParam('optimizer', hp.Discrete(['Adam', 'SGD']))
hparam_domains['mel_bands'] = hp.HParam('mel_bands', hp.Discrete([96]))
hparam_domains['num_frames'] = hp.HParam('num_frames', hp.Discrete([187]))
hparam_domains['samplerate'] = hp.HParam('samplerate', hp.Discrete([16000]))
hparam_domains['frame_size'] = hp.HParam('frame_size', hp.Discrete([512]))
hparam_domains['step_size'] = hp.HParam('step_size', hp.Discrete([256]))
hparam_domains['feature_pipeline'] = hp.HParam('feature_pipeline', hp.Discrete(['essentia', 'tfds']))

METRIC_ACCURACY = 'sparse_categorical_accuracy'
accurary_name = tf.keras.metrics.get(METRIC_ACCURACY).__name__.replace('_', ' ')


def write_hparam_domains(log_dir):
    for f in Path(log_dir).glob('*.hparam_domains.v2'):
        f.unlink(missing_ok=True)
    hparam_metrics = []
    for split_name, split_type in (('validation', hp.Metric.VALIDATION), ('train', hp.Metric.TRAINING)):
        hparam_metrics.append(hp.Metric(f'{split_name}.{METRIC_ACCURACY}', group='', display_name=f'{split_name.title()} {accurary_name.title()}', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.{METRIC_ACCURACY}_std', group='', display_name=f'{split_name.title()} {accurary_name.title()} StDev', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.soft_voting_{METRIC_ACCURACY}', group='', display_name=f'{split_name.title()} Soft Voting {accurary_name.title()}', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.soft_voting_{METRIC_ACCURACY}_std', group='', display_name=f'{split_name.title()} Soft Voting {accurary_name.title()} StDev', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.hard_voting_{METRIC_ACCURACY}', group='', display_name=f'{split_name.title()} Hard Voting {accurary_name.title()}', dataset_type=split_type))
        hparam_metrics.append(hp.Metric(f'{split_name}.hard_voting_{METRIC_ACCURACY}_std', group='', display_name=f'{split_name.title()} Hard Voting {accurary_name.title()} StDev', dataset_type=split_type))
    hparam_metrics.append(hp.Metric('best_epoch', group='', display_name='Best Epoch', dataset_type=hp.Metric.VALIDATION))
    with tf.summary.create_file_writer(str(log_dir), filename_suffix='.hparam_domains.v2').as_default():
        hp.hparams_config(hparams=list(hparam_domains.values()), metrics=hparam_metrics)


def get_features(hparams, train_splits, val_split=None):
    train_ds = tfds.load(f'{hparams["instrument"]}_emotion_recognition', split='+'.join([f'fold{i}' for i in train_splits]), shuffle_files=True, with_info=False, as_supervised=True)
    if val_split is not None:
        val_ds = tfds.load(f'{hparams["instrument"]}_emotion_recognition', split=f'fold{val_split}', shuffle_files=False, with_info=False, as_supervised=True)

    if hparams['feature_pipeline'] == 'essentia':
        def ds_melspectrogram_essentia(ds, frame_size, step_size, num_parallel_calls=tf.data.experimental.AUTOTUNE):
            @tf_datatype_wrapper
            def melspectrogram_essentia(audio, frame_size, step_size):
                audio_input = VectorInput(audio[:, 0])
                fc = FrameCutter(frameSize=int(frame_size), hopSize=int(step_size), startFromZero=True, validFrameThresholdRatio=1)
                extractor = TensorflowInputMusiCNN()
                pool = essentia.Pool()

                audio_input.data >> fc.signal
                fc.frame >> extractor.frame
                extractor.bands >> (pool, "melbands")

                essentia.run(audio_input)
                return pool['melbands']

            def tf_melspectrogram_essentia(audio, frame_size, step_size):
                melbands, = tf.py_function(melspectrogram_essentia, [audio, frame_size, step_size], [tf.float32])
                melbands.set_shape((None, hparams['mel_bands']))
                return melbands

            return ds.map(lambda audio, label: (tf_melspectrogram_essentia(audio, frame_size, step_size), tf.cast(label, tf.int32)), num_parallel_calls)

        train_ds = train_ds.apply(lambda ds: ds_melspectrogram_essentia(ds, hparams['frame_size'], hparams['step_size'], tfdata_parallel))
        if val_split is not None:
            val_ds = val_ds.apply(lambda ds: ds_melspectrogram_essentia(ds, hparams['frame_size'], hparams['step_size'], tfdata_parallel))
    else:
        raise ValueError('Unknown feature type "{}"'.format(hparams['feature_pipeline']))

    if val_split is not None:
        return train_ds, val_ds
    else:
        return train_ds, None
    

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
    sliced_ds = {'train': train_ds.cache(name='cache_presliced_train').apply(lambda ds: ds_step_slicer(ds, hparams['mel_bands'], hparams['num_frames'], -1, tfdata_parallel))}

    train_cardinality = sliced_ds['train'].flat_map(lambda x: x, name='flatten_cardinality').cardinality().numpy()
    if train_cardinality == tf.data.INFINITE_CARDINALITY or train_cardinality == tf.data.UNKNOWN_CARDINALITY or train_cardinality < 0:
        train_cardinality = 3000
    
    train_pipe = sliced_ds['train']\
        .flat_map(lambda x: x, name='flatten_train')\
        .shuffle(train_cardinality, name='shuffle_train')\
        .batch(hparams['batch_size'], num_parallel_calls=tfdata_parallel, name='batch_train')\
        .prefetch(tfdata_parallel, name='prefetch_train')
    if val_ds is not None:
        sliced_ds['validation'] = val_ds.apply(lambda ds: ds_step_slicer(ds, hparams['mel_bands'], hparams['num_frames'], 0, tfdata_parallel)).cache(name='cache_sliced_val')
        val_pipe = sliced_ds['validation']\
            .flat_map(lambda x: x, name='flatten_val')\
            .batch(hparams['batch_size'], num_parallel_calls=tfdata_parallel, name='batch_val')\
            .cache(name='cache_batched_val')\
            .prefetch(tfdata_parallel, name='prefetch_val')
    else:
        val_pipe = None

    log_dir.parent.mkdir(parents=True, exist_ok=True)

    tensorboard = TensorBoard(log_dir=str(log_dir)+log_suffix, profile_batch=0)
    # exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=15, decay_rate=0.4, staircase=True)
    # pcwconst_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay([13, 28, 43, 58], [0.001, 0.0004, 0.00016, 6.4e-5, 4e-5])
    # lrate = LearningRateScheduler(pcwconst_decay, verbose=1)
    # from deepsuite.keras import ConfusionMatrixOnEpoch
    # confusion_matrix = ConfusionMatrixOnEpoch(log_dir, class_names, ['train', 'validation'])
    callbacks = [tensorboard]#[, confusion_matrix, lrate],
    tmp_weights_path = Path(tempfile.gettempdir()) / (exp_name+log_suffix+'.h5')
    if val_ds is not None:
        earlystopper = EarlyStopping(monitor='val_'+METRIC_ACCURACY, patience=hparams['early_stopping_patience'], verbose=1, restore_best_weights=True)
        checkpointer = ModelCheckpoint(tmp_weights_path, monitor='val_'+METRIC_ACCURACY, verbose=1, save_best_only=True)
        callbacks += [earlystopper, checkpointer]

    fit_log = model.fit(train_pipe, validation_data=val_pipe, epochs=hparams['epochs'], verbose=2, callbacks=callbacks)
    if tmp_weights_path.is_file():
        model.load_weights(tmp_weights_path) # when max epochs gets exceeded, early stopping callback doesn't load best model
        tmp_weights_path.unlink()
    if save_model_dir:
        save_path = save_model_dir / (exp_name+log_suffix+'.h5')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.get_layer('frontend').trainable = True
        model.get_layer('midend').trainable = True
        model.get_layer('backend').bn_flat_pool.trainable = True
        model.get_layer('backend').penultimate.trainable = True
        model.get_layer('backend').bn_penultimate.trainable = True
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
    with tf.summary.create_file_writer(str(log_dir), filename_suffix='.final.v2').as_default():
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
                    print(f'The final model has achieved a {split_name} {metric_name} of {value[0]:.3f} +/- {value[1]:.3f}')
                except TypeError:
                    tf.summary.scalar(f'{split_name}.{metric_name}', value, step=0)
                    print(f'The final model has achieved a {split_name} {metric_name} of {value:.3f} at epoch {best_epoch+1}')

        for split_name, conf in conf_mat.items():
            tf.summary.image(f'{split_name.title()} Confusion', mpl_fig_to_tf_image(plot_confusion_matrix(conf, class_names, normalize=True, title='')), step=best_epoch)

        for split_name, (soft_metric, hard_metric) in majority_eval_results.items():
            try:
                tf.summary.scalar(f'{split_name}.soft_voting_{METRIC_ACCURACY}', soft_metric[0], step=0)
                tf.summary.scalar(f'{split_name}.soft_voting_{METRIC_ACCURACY}_std', soft_metric[1], step=0)
                tf.summary.scalar(f'{split_name}.hard_voting_{METRIC_ACCURACY}', hard_metric[0], step=0)
                tf.summary.scalar(f'{split_name}.hard_voting_{METRIC_ACCURACY}_std', hard_metric[1], step=0)
                print(f'The final model has achieved a {split_name} soft voting {accurary_name} of {100*soft_metric[0]:.3f}% +/- {100*soft_metric[1]:.3f}% and a {split_name} hard voting {accurary_name} of {100*hard_metric[0]:.3f}% +/- {100*hard_metric[1]:.3f}%')
            except IndexError:
                tf.summary.scalar(f'{split_name}.soft_voting_{METRIC_ACCURACY}', soft_metric, step=0)
                tf.summary.scalar(f'{split_name}.hard_voting_{METRIC_ACCURACY}', hard_metric, step=0)
                print(f'The final model has achieved a {split_name} soft voting {accurary_name} of {100*soft_metric:.3f}% and a {split_name} hard voting {accurary_name} of {100*hard_metric:.3f}% at epoch {best_epoch+1}')
        
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

    ds_info = tfds.builder(f'{hparams["instrument"]}_emotion_recognition').info
    class_names = ds_info.features['emotion'].names
    num_folds = len(ds_info.splits)

    exp_name = str(Path(exp_base_name) / datetime.now().strftime("%y%m%d-%H%M%S"))
    log_dir = log_base_dir / exp_name

    if hparams['num_folds'] == num_folds:
        fold_outputs = []
        if not hparams['finetuning']:
            init_model = get_model(hparams, len(class_names))
        for fold_idx in range(1, num_folds+1):
            print(f'\nFold {fold_idx}')
            train_splits = [t for t in range(1, num_folds+1) if t != fold_idx]
            train_ds, val_ds = get_features(hparams, train_splits, fold_idx)
            fold_suffix = f'/fold{fold_idx}'
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
    else:
        train_ds, val_ds = get_features(hparams, range(num_folds))
        model = get_model(hparams, len(class_names))
        model_output = fit_model(model, exp_name, log_dir, save_model_dir, train_ds, val_ds)
        write_log(log_dir, hparams, exp_name, class_names, model.metrics_names, *model_output)




if __name__ == '__main__':
    
    import argparse
    import itertools
    from distutils.util import strtobool
    
    log_base_dir = Path('./tensorboard')
    exp_base_name = Path(__file__).stem

    def list_saved_models(save_model_dir, exp_base_name):
        d = save_model_dir / exp_base_name
        try:
            return [str('..' / x) for x in d.iterdir() if x.is_dir() or x.suffix == '.h5']
        except FileNotFoundError:
            return []

    hparam_domains['weights'] = hp.HParam('weights', hp.Discrete(hparam_domains['weights'].domain.values + list_saved_models(Path('saved-models'), exp_base_name)))

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
    Run instrument emotion recognition experiment.''',
    allow_abbrev=False)
    features_config = parser.add_argument_group('Feature options')
    features_config.add_argument('-i', '--instrument', default=['acoustic_guitar'], action=ResetAppendAction, type=str, choices=hparam_domains['instrument'].domain.values)
    features_config.add_argument('-m', '--mel-bands', default=[96], action=ResetAppendAction, type=int)
    features_config.add_argument('-n', '--num-frames', default=[187], action=ResetAppendAction, type=int)
    features_config.add_argument('-r', '--samplerate', default=[16000], action=ResetAppendAction, type=int)
    features_config.add_argument('-f', '--frame-size', default=[512], action=ResetAppendAction, type=int)
    features_config.add_argument('-s', '--step-size', default=[256], action=ResetAppendAction, type=int)
    features_config.add_argument('-t', '--feature-pipeline', default=['essentia'], action=ResetAppendAction, type=str, choices=hparam_domains['feature_pipeline'].domain.values)

    model_config = parser.add_argument_group('Model options')
    model_config.add_argument('-c', '--classifier-activation', default=['relu'], action=ResetAppendAction, type=str, choices=hparam_domains['classifier_activation'].domain.values)
    model_config.add_argument('-a', '--final-activation', default=['softmax'], action=ResetAppendAction, type=str, choices=hparam_domains['final_activation'].domain.values)
    model_config.add_argument('-w', '--weights', default=['MTT_musicnn'], action=ResetAppendAction, type=str, choices=hparam_domains['weights'].domain.values)

    train_config = parser.add_argument_group('Training options')
    train_config.add_argument('--num-folds', default=[5], action=ResetAppendAction, type=int)
    train_config.add_argument('--tf-seed', default=[0], action=ResetAppendAction, type=int)
    train_config.add_argument('-o', '--optimizer', default=['Adam'], action=ResetAppendAction, type=str, choices=hparam_domains['optimizer'].domain.values)
    train_config.add_argument('-l', '--learning-rate', default=[0.0001], action=ResetAppendAction, type=float)
    train_config.add_argument('-b', '--batch-size', default=[256], action=ResetAppendAction, type=int)
    train_config.add_argument('-e', '--epochs', default=[100], action=ResetAppendAction, type=int)
    train_config.add_argument('--early-stopping-patience', default=[30], action=ResetAppendAction, type=int)
    train_config.add_argument('--save-model-dir', default='', action='store', type=str)
    train_config.add_argument('--finetuning', default=[False], action=ResetAppendAction, type=lambda x: bool(strtobool(x)))

    args = vars(parser.parse_args())
    save_model_dir = Path(args.pop('save_model_dir'))
    write_hparam_domains(log_base_dir / exp_base_name)

    for param_combo in itertools.product(*args.values()):
        hparams = dict(zip(args.keys(), param_combo))
        print(f'Running parameter combination {", ".join(["{}: {}".format(k, v) for k, v in hparams.items()])}')
        run_experiment(hparams, log_base_dir, exp_base_name, save_model_dir)
