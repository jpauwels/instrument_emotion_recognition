from .plotting import mpl_fig_to_tf_image

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

import os.path


class ModelWithBatchLabels(tf.keras.models.Model):
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
#         _minimize(self.distribute_strategy, tape, self.optimizer, loss,
#                   self.trainable_variables)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # return {'metrics': {m.name: m.result() for m in self.metrics}, 'labels': {'true': y, 'pred': y_pred}}
        return {**{m.name: m.result() for m in self.metrics}, 'y': y, 'y_pred': y_pred}
        
    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # return {'metrics': {m.name: m.result() for m in self.metrics}, 'labels': {'true': y, 'pred': y_pred}}
        return {**{m.name: m.result() for m in self.metrics}, 'y': y, 'y_pred': y_pred}
        

class ConfusionMatrixOnEpoch(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, class_names, splits={'train', 'validation'}, normalize=False):
        self.file_writers = {split: tf.summary.create_file_writer(os.path.join(log_dir, split), filename_suffix='.confmat.v2') for split in splits}
        self.class_names = class_names
        self.splits = splits
        self.normalize = normalize
        
    def on_epoch_begin(self, epoch, logs):
        self.true_labels = {split: [] for split in self.splits}
        self.pred_labels = {split: [] for split in self.splits}
        
    def on_train_batch_end(self, batch, logs):
        if 'train' in self.splits:
            # self.true_labels['train'].append(logs['labels']['true'])
            # self.pred_labels['train'].append(logs['labels']['pred'])
            self.true_labels['train'].append(logs['y'])
            self.pred_labels['train'].append(logs['y_pred'])
        
    def on_test_batch_end(self, batch, logs):
        if 'validation' in self.splits:
            # self.true_labels['validation'].append(logs['labels']['true'])
            # self.pred_labels['validation'].append(logs['labels']['pred'])
            self.true_labels['validation'].append(logs['y'])
            self.pred_labels['validation'].append(logs['y_pred'])
        
    def on_epoch_end(self, epoch, logs):
        # print(len(logs['labels']['true']), len(logs['labels']['pred']))
        for split in self.splits:
            true_labels_raw = tf.concat(self.true_labels[split], axis=0)
            true_labels = tf.argmax(true_labels_raw, axis=1)
            pred_labels_raw = tf.concat(self.pred_labels[split], axis=0)
            pred_labels = tf.argmax(pred_labels_raw, axis=1)
            conf_mat = tf.math.confusion_matrix(true_labels, pred_labels)
            fig = plot_confusion_matrix(conf_mat, self.class_names, normalize=self.normalize)

            with self.file_writers[split].as_default():
                tf.summary.image('Confusion Matrix', mpl_fig_to_tf_image(fig), step=epoch+1, description='{} confusion matrix'.format(split))
                
    def on_train_end(self, logs):
        for file_writer in self.file_writers.values():
            file_writer.close()


# class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
#     def __init__(self, log_dir, class_names, data, normalize=True):
#         super().__init__()
#         self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'confusion-matrices'))
#         self.class_names = class_names
#         self.data = data
#         self.normalize = normalize
        
#     def on_epoch_end(self, epoch, logs):
#         for data_name, data_pipe in self.data.items():
#             features, true_labels = [tf.concat(x, axis=0) for x in zip(*data_pipe)]
#             data_pred = model.predict(features, verbose=0)
#             pred_labels = tf.argmax(data_pred, axis=1)
#             conf_mat = tf.math.confusion_matrix(true_labels, pred_labels)
#             fig = plot_confusion_matrix(conf_mat, self.class_names, normalize=self.normalize)

#             with self.file_writer.as_default():
#                 tf.summary.image("{} Confusion Matrix".format(data_name), mpl_fig_to_tf_image(fig), step=epoch)