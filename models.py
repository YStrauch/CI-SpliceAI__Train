import keras
import numpy as np
import env
import tensorflow as tf

def conv(val, filters, size, dilation):
    val = keras.layers.BatchNormalization()(val)
    val = keras.layers.Activation('relu')(val)
    return keras.layers.Conv1D(filters, size, dilation_rate=dilation, padding='same')(val)


def skip(inputs, filters, size, dilation):
    val = inputs
    for _ in range(2):
        val = conv(val, filters, size, dilation)

    return keras.layers.add([val, inputs])

def skip_block(val, filters, size, dilation):
    for _ in range(4):
        val = skip(val, filters, size, dilation)

    return val
        

def CNN():
    inputs = keras.layers.Input(shape=(None, 4))

    # main strand -- the skip modules (4x2 convs)
    # skip strand -- the skip connection consisting of conv layers

    main_strand = keras.layers.Conv1D(32, 1, dilation_rate=1, padding='same')(inputs)
    skip_strand = keras.layers.Conv1D(32, 1, dilation_rate=1, padding='same')(main_strand)

    for window_size, dilation in zip([11, 11, 21, 41], [1, 4, 10, 25]):
        main_strand = skip_block(main_strand, 32, window_size, dilation)
        skip_strand = keras.layers.add([keras.layers.Conv1D(32, 1, dilation_rate=1, padding='same')(main_strand), skip_strand])

    crop_len = int(env.CONTEXT_LEN // 2)
    crop = keras.layers.Cropping1D((crop_len, crop_len))
    shape_conv = keras.layers.Conv1D(3, 1, dilation_rate=1, padding='same', activation='softmax')
    output = shape_conv(crop(skip_strand))

    return keras.models.Model(inputs=inputs, outputs=output)


def DataParallelModel(model, gpus):
    
    def slice_data(x, n_slices, slice_index):
        gpu_slice_len = tf.shape(x)[0] // n_slices
        if slice_index == n_slices - 1:
            # consume all remaining to make up for rounding errors
            return x[slice_index * gpu_slice_len:]
        # slice batch for this GPU
        return x[slice_index * gpu_slice_len:(slice_index+1)*gpu_slice_len]
    
    outputs = []

    for device_index, device in enumerate(gpus):
        with tf.device(device):
            input_gpu = []

            for x in model.inputs:
                x_slice = keras.layers.core.Lambda(slice_data, arguments={'n_slices': len(gpus), 'slice_index': device_index})(x)
                input_gpu.append(x_slice)

            outputs.append(model(input_gpu))

    with tf.device('/cpu:0'):
        outputs_cpu = keras.layers.concatenate(outputs, axis=0)
            
        return keras.models.Model(inputs=model.inputs, outputs=[outputs_cpu])

if __name__ == '__main__':
    model = CNN()

    print(model.summary())