'''Script for training a mask classification model.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os
import h5py
import horovod.tensorflow as hvd
# hvd.init()
# os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import itertools
# from skimage.segmentation import find_boundaries
# from ffn.utils import bounding_box
# from ffn.training import inputs
from ffn.training.import_util import import_symbol
from ffn_mask import io_utils
from ffn_mask import model_utils

import sys
from mpi4py import MPI
import json

# tf.disable_v2_behavior()

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
flags.DEFINE_string('label_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing int64 '
                    'label data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('weights_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing int64 '
                    'label data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('tf_coords', None, 
                    'Prefix to tfrecord files with coordinates')
flags.DEFINE_string('train_dir', None, '')
flags.DEFINE_string('model_name', None,
                    'Name of the model to train. Format: '
                    '[<packages>.]<module_name>.<model_class>, if packages is '
                    'missing "ffn.training.models" is used as default.')
flags.DEFINE_string('model_args', None,
                    'JSON string with arguments to be passed to the model '
                    'constructor.')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_float('image_mean', 128, '')
flags.DEFINE_float('image_stddev', 33, '')
flags.DEFINE_integer('max_steps', 100000, '')
flags.DEFINE_boolean('rotation', False, '')



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
  num_classes = int(model_args['num_classes'])

  if num_classes == 1:
    model_fn = model_utils.mask_model_fn_regression  
    # model_fn = model_utils.mask_model_fn_classfication
  else:
    model_fn = model_utils.mask_model_fn_classfication

  params = {
    'model_class': model_class,
    'model_args': model_args,
    'batch_size': FLAGS.batch_size,
    'num_classes': num_classes,
    'learning_rate': FLAGS.learning_rate
  }

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
  sess_config = tf.compat.v1.ConfigProto()
  # sess_config = tf.config()
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

  model_dir = FLAGS.train_dir if hvd.rank() == 0 else None
  save_summary_steps = 90 if hvd.rank() == 0 else None
  save_checkpoints_secs = 540 if hvd.rank() == 0 else None

  config=tf.estimator.RunConfig( 
    model_dir=model_dir,
    save_summary_steps=save_summary_steps,
    save_checkpoints_secs=save_checkpoints_secs,
    session_config=sess_config,
    keep_checkpoint_max=10,
  )
  mask_estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=config,
    params=params
    )
  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
  # logging_hook = tf.estimator.LoggingTensorHook(
  # # logging_hook = tf.train.LoggingTensorHook(
  #   {'flat_logits': 'flat_logits',
  #    'flat_labels': 'flat_labels'},
  #   #  'flat_weights': 'flat_weights'},
  #    every_n_iter=100,
  # )

  if FLAGS.weights_volumes:
    input_fn = io_utils.train_input_fn_with_weight(
      FLAGS.data_volumes, 
      FLAGS.label_volumes, 
      FLAGS.weights_volumes,
      FLAGS.tf_coords,
      num_classes,
      fov_size, 
      label_size,
      FLAGS.batch_size, 
      FLAGS.image_mean, 
      FLAGS.image_stddev,
      FLAGS.rotation)
  else:
    input_fn = io_utils.train_input_fn(
      FLAGS.data_volumes, 
      FLAGS.label_volumes, 
      FLAGS.tf_coords,
      num_classes,
      fov_size, 
      label_size,
      FLAGS.batch_size, 
      FLAGS.image_mean, 
      FLAGS.image_stddev,
      FLAGS.rotation)

  mask_estimator.train(
    input_fn=input_fn,
    steps=FLAGS.max_steps,
    hooks=[bcast_hook])
    # hooks=[bcast_hook, logging_hook])

if __name__ == '__main__':
  app.run(main)


