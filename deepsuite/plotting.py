import io
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_metrics(history, metrics_names=None):
    if not metrics_names:
        metrics_names = filter(lambda m: not m.startswith('val_'), history.keys())
    for metric in metrics_names:
        fig, ax = plt.subplots()
        ax.plot(history[metric])
        if 'val_'+metric in history:
            ax.plot(history['val_'+metric])
            ax.legend(['train', 'val'])
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        fig.show()
        
        
# Adapted from https://www.tensorflow.org/tensorboard/image_summaries
def plot_confusion_matrix(conf_mat, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        conf_mat, _ = tf.linalg.normalize(conf_mat, ord=1, axis=1)
        conf_mat *= 100
        vmax = 100
    else:
        vmax = tf.reduce_max(tf.reduce_sum(conf_mat, axis=1))#.numpy()

    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, format='%.1f' if normalize else '%d')
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = tf.cast(tf.reduce_max(conf_mat) / 2, conf_mat.dtype)
    fmt = '.1f' if normalize else 'd'
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        ax.text(j, i, format(conf_mat[i, j], fmt)+('%' if normalize else ''), size=11,
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    ax.set_title(title, fontsize=16)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    fig.set_tight_layout(True)
    plt.ion()
    return fig


# Adapted from https://www.tensorflow.org/tensorboard/image_summaries
def mpl_fig_to_tf_image(fig):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(fig)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.io.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image