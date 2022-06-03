import keras
import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def DataParallelModel(model, gpus):
    
    def gpu_slice(x, n_slices, slice_index):
        gpu_slice_len = tf.shape(x)[0] // n_slices
        return x[slice_index*gpu_slice_len:(slice_index+1)*gpu_slice_len]
    
    outputs = []

    for device_index, device in enumerate(gpus):
        with tf.device(device):
            input_gpu = []

            for x in model.inputs:
                x_slice = keras.layers.core.Lambda(gpu_slice, arguments={'n_slices': len(gpus), 'slice_index': device_index})(x)
                input_gpu.append(x_slice)

            outputs.append(model(input_gpu))

    with tf.device('/cpu:0'):
        outputs_cpu = keras.layers.concatenate(outputs, axis=0)
            
        return keras.models.Model(inputs=model.inputs, outputs=[outputs_cpu])
