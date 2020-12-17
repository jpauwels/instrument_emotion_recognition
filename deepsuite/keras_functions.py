from .plotting import plot_confusion_matrix
import tensorflow as tf


def get_pred_labels(model, labelled_pairs):
    features, true_labels = [tf.concat(x, axis=0) for x in zip(*labelled_pairs)]
    pred = model.predict(features, verbose=0)
    pred_labels = tf.argmax(pred, axis=1)
    return true_labels, pred_labels
    
def get_confusion_matrix(model, labelled_pairs, classes, normalize=False, title=''):
    conf_mat = tf.math.confusion_matrix(*get_pred_labels(model, labelled_pairs))
    return plot_confusion_matrix(conf_mat, classes, normalize, title=title)