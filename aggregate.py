'''
Aggregates and freezes models/ALL/<1-5>.h5 into models/ALL/ensemble_frozen.pb
'''

import tensorflow as tf
import keras
import os
import argparse
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    '''Freezes the state of a session into a pruned computation graph.'''
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def freeze_and_optimise_ensemble(models, folder, output):
    models = [keras.models.load_model(os.path.join(folder, f'{i}.h5')) for i in models]
    average_ensemble = keras.layers.Lambda(lambda x: keras.backend.mean(keras.backend.stack([model(x) for model in models]), axis=0), name='output')(models[0].input)
    ensemble = keras.models.Model(inputs=models[0].inputs, outputs=average_ensemble, name='CI-SpliceAI')

    frozen = freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in ensemble.outputs])

    # optimise for inference
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        frozen,
        [frozen.node[0].name], # hard coding one input node
        [frozen.node[-1].name], # hard coding one output node
        dtypes.float32.as_datatype_enum
    )

    # output graph
    with gfile.FastGFile(output, 'w') as f:
        f.write(output_graph_def.SerializeToString())

    print('Ensemble inputs: %s; outputs: %s' % (','.join([i.name for i in ensemble.inputs]), ','.join([i.name for i in ensemble.outputs])))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freeze and optimise Ensemble')
    parser.add_argument('--models', default="1,2,3,4,5", help='Comma-separated trained model indices; defaults to "1,2,3,4,5"')
    parser.add_argument('--folder', default=os.path.join('models', 'ALL'), help='Path to folder containing the models; defaults to "models/ALL"')
    parser.add_argument('--output', default=None, help='Path to the the output .pb file. Defaults to "<folder>/CI-Spliceai.pb"')

    args = parser.parse_args()

    models = args.models.split(',')
    folder = args.folder
    output = args.output if args.output is not None else os.path.join(folder, 'CI-SpliceAI.pb')

    keras.backend.set_learning_phase(0)
    freeze_and_optimise_ensemble(models=models, folder=folder, output=output)