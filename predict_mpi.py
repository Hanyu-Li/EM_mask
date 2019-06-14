'''Script for predicting with a mask classification model.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import h5py
import tensorflow as tf
import numpy as np
import itertools
from skimage.segmentation import find_boundaries
from ffn.utils import bounding_box
from ffn.training import inputs
from ffn.training.import_util import import_symbol
from ffn_mask import io_utils, model_utils

import horovod.tensorflow as hvd
import sys
# from mpi4py import MPI
import json

# comm = MPI.COMM_WORLD
# rank = comm.rank

FLAGS = flags.FLAGS

flags.DEFINE_string('data_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing uint8 '
                    'image data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('output_volumes', None, '')
flags.DEFINE_string('model_checkpoint', None, '')  
flags.DEFINE_string('model_name', None,
                    'Name of the model to train. Format: '
                    '[<packages>.]<module_name>.<model_class>, if packages is '
                    'missing "ffn.training.models" is used as default.')
flags.DEFINE_string('model_args', None,
                    'JSON string with arguments to be passed to the model '
                    'constructor.')

flags.DEFINE_string('bounding_box', None, '')
flags.DEFINE_float('var_threshold', 0, '')
flags.DEFINE_list('overlap', None, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_float('image_mean', 128, '')
flags.DEFINE_float('image_stddev', 33, '')
flags.DEFINE_integer('max_steps', 100000, '')
flags.DEFINE_boolean('mpi', False, '')


def main(unused_argv):
  hvd.init()
  model_class = import_symbol(FLAGS.model_name, 'ffn_mask')
  model_args = json.loads(FLAGS.model_args)
  fov_size= tuple([int(i) for i in model_args['fov_size']])
  if 'label_size' in model_args:
    label_size = tuple([int(i) for i in model_args['label_size']])
  else:
    label_size = fov_size
    model_args['label_size'] = label_size
  overlap = [int(i) for i in FLAGS.overlap]
  num_classes = int(model_args['num_classes'])
  params = {
    'model_class': model_class,
    'model_args': model_args,
    'batch_size': FLAGS.batch_size,
    'num_classes': num_classes
  }

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.visible_device_list = str(hvd.local_rank())


  if num_classes == 1:
    model_fn = model_utils.mask_model_fn_regression  
  else:
    model_fn = model_utils.mask_model_fn_classfication

  model_checkpoint = FLAGS.model_checkpoint if hvd.rank() == 0 else None

  config=tf.estimator.RunConfig( 
    session_config=sess_config,
  )
  mask_estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=config,
    params=params,
    warm_start_from=model_checkpoint
  )
  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
  tensors_to_log = {
    "center": "center"
  }
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=1
  )


  # it=io_utils.predict_input_fn_v2(
  #   data_volumes=FLAGS.data_volumes, 
  #   chunk_shape=fov_size, 
  #   label_shape=label_size,
  #   overlap=overlap,
  #   batch_size=FLAGS.batch_size, 
  #   offset=FLAGS.image_mean, 
  #   scale=FLAGS.image_stddev,
  #   sub_bbox=FLAGS.bounding_box,
  #   var_threshold=FLAGS.var_threshold)
  
  # with tf.Session() as sess:
  #   for i in range(3):
  #     res = sess.run(it)
  #     print(res[0])
      #print(res[0], res[1].shape, np.mean(res[1]))




  predictions = mask_estimator.predict(
    input_fn=lambda: io_utils.predict_input_fn_v2(
      data_volumes=FLAGS.data_volumes, 
      chunk_shape=fov_size, 
      label_shape=label_size,
      overlap=overlap,
      batch_size=FLAGS.batch_size, 
      offset=FLAGS.image_mean, 
      scale=FLAGS.image_stddev,
      sub_bbox=FLAGS.bounding_box,
      var_threshold=FLAGS.var_threshold),
    predict_keys=['center', 'logits', 'class_prediction'],
    hooks=[logging_hook, bcast_hook],
    yield_single_examples=True
  )
  output_shapes = io_utils.get_h5_shapes(FLAGS.data_volumes)
  io_utils.h5_sequential_chunk_writer_v2(
    predictions,
    output_volumes=FLAGS.output_volumes,
    output_shapes=output_shapes,
    num_classes=num_classes,
    chunk_shape=fov_size,
    label_shape=label_size,
    overlap=overlap,
    sub_bbox=FLAGS.bounding_box,
    mpi=FLAGS.mpi)


if __name__ == '__main__':
  app.run(main)


