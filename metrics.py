import keras.backend as kb
import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score


def categorical_crossentropy(target, output, epsilon=10e-8):
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return - kb.mean(
        target[:, :, 0] * kb.log(output[:, :, 0]) +
        target[:, :, 1] * kb.log(output[:, :, 1]) +
        target[:, :, 2] * kb.log(output[:, :, 2])
    )


def average_precision_multiclass_numpy(y_true, y_pred, classes=(1,2)):
    score = 0.0

    for site_type in classes:
        yt = y_true[:, :, site_type].flatten()
        if (yt == False).all():
            # edge case: only negative labels, so add 100% for this class, see https://github.com/scikit-learn/scikit-learn/issues/8245
            score += 1/len(classes)
        else:
            yp = y_pred[:, :, site_type].flatten()
            mask = np.logical_and(np.isfinite(yt), np.isfinite(yp))
            if (mask == True).any():
                score += average_precision_score(yt[mask], yp[mask])/len(classes)

    return score

def average_precision_multiclass(y_true, y_pred, classes=(1,2)):
    return tf.py_func(average_precision_multiclass_numpy, (y_true, y_pred, classes), tf.double)