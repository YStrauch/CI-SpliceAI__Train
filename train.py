'''
Trains the CNN model.

Usage:
train.py <model_index> <fold>

where <model_index> is a number between 1 and 5
and <fold> is either ALL or TEST; ALL trains on all chromosomes and TEST excludes the test data (see TEST_FOLD in env.py for a list of test chromosomes)
'''

import numpy as np
import sys
import os
import h5py
import keras
from tensorflow.python.client import device_lib
import models
import env
import metrics
import splicing_table

# -- SCRIPT PARAMS --
# default to model index 0
if len(sys.argv) < 2:
    sys.argv.append(0)
# default to training on all chromosomes
if len(sys.argv) < 3:
    sys.argv.append('ALL')

model_number = int(sys.argv[1])
fold_name = sys.argv[2].upper()
fold = env.FOLDS[fold_name]
assert fold != env.FOLDS['TEST'], "Must not train on TEST fold"

# -- OUTPUT FOLDERS --
model_folder = os.path.join('models', fold_name)
log_folder = os.path.join('logs', fold_name, str(model_number))
os.makedirs(model_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# -- MULTI-GPU --
local_device_protos = device_lib.list_local_devices()
gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
if len(gpus) == 0:
    print('WARNING: NO GPUS FOUND, DEFAULT TO CPU. THIS SHOULD NOT HAPPEN IF YOU ACTUALLY WANT TO TRAIN!')
    gpus = ['/cpu:0']
print(f'Training on {",".join(gpus)}')

# -- MODEL --
model = models.CNN()
if len(gpus) > 1:
    model_distributed = models.DataParallelModel(model, gpus)
else:
    model_distributed = model
model_distributed.compile(loss=metrics.categorical_crossentropy, optimizer='adam', metrics = [metrics.average_precision_multiclass])

# -- DATA --
n_chunks, splice_table = splicing_table.get_train_splice_table(fold)

# use 10% of chunks as validation data
validate_chunks = np.random.choice(range(n_chunks), int(n_chunks * env.VALIDATE_SPLIT_SIZE), replace=False)
train_chunks = list(set(range(n_chunks)) - set(validate_chunks))
assert set(validate_chunks).isdisjoint(train_chunks)

train_data_size = splice_table[splice_table.chunk.isin(train_chunks)].slices_in_gene.sum()
val_data_size = splice_table[splice_table.chunk.isin(validate_chunks)].slices_in_gene.sum()

data = h5py.File(env.TRAIN_FILE, 'r')
if fold == env.FOLDS['ALL']:
    assert len(data['X']) == train_data_size + val_data_size

def get_validate_chunk():
    while True:
        for chunk in validate_chunks:
            yield splicing_table.slices_in_chunk(splice_table, data, len(gpus), chunk)

def get_train_chunk():
    while True:
        yield splicing_table.slices_in_chunk(splice_table, data, len(gpus), np.random.choice(train_chunks))

train_data_it = get_train_chunk()
val_data_it = get_validate_chunk()

# -- TRAIN --
n_epochs = fold.train_iterations * len(train_chunks)
batch_size = env.BATCH_SIZE_PER_GPU * len(gpus)

tensorboard = keras.callbacks.TensorBoard(
    log_dir=log_folder
)

# learning rate
reduce_lr_at = [int(completion * n_epochs) for completion in env.LEARNING_RATE_DECAY_AT]
print(f'Reduce LR after {",".join(map(str, reduce_lr_at))} epochs with factor {env.LEARNING_RATE_DECAY_FACTOR}')
lr = env.LEARNING_RATE_START
def lr_schedule(epoch):
    global lr
    
    if epoch in reduce_lr_at:
        lr *= env.LEARNING_RATE_DECAY_FACTOR

    return lr
learning_rate_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)


# ideally we would fit using the two generators directly, but this keras version doesn't allow specifying steps per epoch with generators
# instead, we iterate over the epochs manually
for epoch_num in range(n_epochs):
    model_distributed.fit(
        *next(train_data_it),
        verbose=2,
        callbacks=[
            tensorboard,
            learning_rate_scheduler
        ],
        epochs=epoch_num+1,
        initial_epoch=epoch_num,
        validation_data=next(val_data_it),
        batch_size=batch_size
    )

data.close()
model.save(os.path.join(model_folder, '%s.h5' % model_number))
