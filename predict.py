'''Script for training a mask classification model.'''
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
from ffn_mask import utils as mask_utils

import horovod.tensorflow as hvd
import sys
from mpi4py import MPI
import json

comm = MPI.COMM_WORLD
rank = comm.rank

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

  # model_dir = FLAGS.train_dir if hvd.rank() == 0 else None
  # save_summary_steps = 30 if hvd.rank() == 0 else None
  # save_checkpoints_secs = 30 if hvd.rank() == 0 else None

  model_checkpoint = FLAGS.model_checkpoint if hvd.rank() == 0 else None

  config=tf.estimator.RunConfig( 
    # model_dir=model_dir,
    # save_summary_steps=save_summary_steps,
    # save_checkpoints_secs=save_checkpoints_secs,
    session_config=sess_config,
    # keep_checkpoint_max=1000,
  )
  mask_estimator = tf.estimator.Estimator(
    model_fn=mask_utils.mask_model_fn,
    config=config,
    params=params,
    warm_start_from=model_checkpoint
  )
  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
  # mask_estimator.train(
  #   input_fn = lambda: mask_input_fn(
  #     FLAGS.data_volumes, 
  #     FLAGS.label_volumes, 
  #     fov_size, 
  #     FLAGS.batch_size, 
  #     FLAGS.image_mean, 
  #     FLAGS.image_stddev),
  #   steps=FLAGS.max_steps,
  #   hooks=[bcast_hook])
  tensors_to_log = {
    "center": "center"
  }
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=1
  )
  predictions = mask_estimator.predict(
    input_fn=lambda: mask_utils.predict_input_fn(
      data_volumes=FLAGS.data_volumes, 
      chunk_shape=fov_size, 
      overlap=overlap,
      batch_size=FLAGS.batch_size, 
      offset=FLAGS.image_mean, 
      scale=FLAGS.image_stddev,
      bounding_box=FLAGS.bounding_box,
      var_threshold=FLAGS.var_threshold),
    predict_keys=['center', 'class_prediction'],
    hooks=[logging_hook, bcast_hook],
    # checkpoint_path=FLAGS.model_checkpoint,
    yield_single_examples=True
  )
  output_shapes = mask_utils.get_h5_shapes(FLAGS.data_volumes)
  mask_utils.h5_sequential_chunk_writer(predictions,
    output_volumes=FLAGS.output_volumes,
    output_shapes=output_shapes,
    num_classes=num_classes,
    chunk_shape=fov_size,
    overlap=overlap,
    mpi=FLAGS.mpi)

  # f_w = h5py.File('')

  # for i in results:
  #   pass
    # print(hvd.rank(), i['center'])


if __name__ == '__main__':
  app.run(main)


