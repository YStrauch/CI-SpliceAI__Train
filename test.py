'''
Tests the 5 CNN models.

Usage:
test.py

Assumes they were trained on TRAIN split with model indices 1-5.
'''

import numpy as np
import os
import h5py
import keras
import env
import metrics
import splicing_table
from lib.progress import ProgressOutput

keras.backend.set_learning_phase(0)
model_folder = os.path.join('models', 'TRAIN')
model_indices = list(range(1,6))

# -- DATA --
n_chunks, splice_table = splicing_table.get_train_splice_table(env.FOLDS['TEST'], drop_paralogs=True)
data_size = splice_table.slices_in_gene.sum()
data = h5py.File(env.TRAIN_FILE, 'r')
prog = ProgressOutput(n_chunks)

def get_test_chunk():
    for chunk in range(n_chunks):
        yield splicing_table.slices_in_chunk(splice_table, data, 1, chunk)

# -- MODELS --
models = [keras.models.load_model(os.path.join(model_folder, f'{i}.h5')) for i in model_indices]
batch_size = env.BATCH_SIZE_PER_GPU

# -- PREDICT --
Y_true = []
Y_pred = []

for i, (X, Y) in enumerate(get_test_chunk()):
    Y_true += list(Y[0])
    Y_pred += list(np.mean([model.predict(X, batch_size=env.BATCH_SIZE_PER_GPU) for model in models], axis=0))

    prog.update(i+1)

data.close()

Y_true = np.array(Y_true)
Y_pred = np.array(Y_pred)

# -- MEAN AVERAGE PRECISION SCORE --
print('MEAN AVERAGE PRECISION SCORE:')
print(f'Acc.: \t{np.mean(metrics.average_precision_multiclass_numpy(Y_true, Y_pred, classes=(1,)))}')
print(f'Don.: \t{np.mean(metrics.average_precision_multiclass_numpy(Y_true, Y_pred, classes=(2,)))}')
print(f'Both: \t{np.mean(metrics.average_precision_multiclass_numpy(Y_true, Y_pred, classes=(1,2)))}')