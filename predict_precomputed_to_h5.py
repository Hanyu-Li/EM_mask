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
# from ffn_mask import io_utils, model_utils
from ffn_mask import precomputed_utils, model_utils


# import horovod.tensorflow as hvd
import sys
import json

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


FLAGS = flags.FLAGS

flags.DEFINE_string('input_volume', None,
                    'Path to precomputed volume')
flags.DEFINE_string('input_offset', '',
                    'offset x,y,z')
flags.DEFINE_string('input_size', '',
                    'size x,y,z')
flags.DEFINE_integer('input_mip', 0,
                    'mip level to read from cloudvolume')
flags.DEFINE_string('output_volume', None, '')
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
flags.DEFINE_list('use_gpu', [], '')


def prepare_model(model_params, model_checkpoint, use_gpu=[]):
  if not len(use_gpu):
    sess_config = tf.ConfigProto(
      device_count={'GPU': 0}
    )
  else:
    rank_gpu = str(mpi_rank % len(use_gpu))
    gpu_options = tf.GPUOptions(visible_device_list=rank_gpu, allow_growth=True)
    sess_config = tf.ConfigProto(
      gpu_options=gpu_options
    )

  # sess_config.gpu_options.allow_growth = True
  # sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

  if model_params['num_classes'] == 1:
    model_fn = model_utils.mask_model_fn_regression  
  else:
    model_fn = model_utils.mask_model_fn_classfication

  # model_checkpoint = FLAGS.model_checkpoint if hvd.rank() == 0 else None
  model_checkpoint = FLAGS.model_checkpoint

  config=tf.estimator.RunConfig( 
    session_config=sess_config,
  )
  mask_estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=config,
    params=model_params,
    warm_start_from=model_checkpoint
  )
  # bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

  return mask_estimator



def main(unused_argv):
  # hvd.init()
  model_class = import_symbol(FLAGS.model_name, 'ffn_mask')
  model_args = json.loads(FLAGS.model_args)
  fov_size= tuple([int(i) for i in model_args['fov_size']])

  if FLAGS.input_offset and FLAGS.input_size:
    input_offset= np.array([int(i) for i in FLAGS.input_offset.split(',')])
    input_size= np.array([int(i) for i in FLAGS.input_size.split(',')])
  else:
    # input_offset = None
    # input_size = None
    input_offset, input_size = precomputed_utils.get_offset_and_size(FLAGS.input_volume)

  if 'label_size' in model_args:
    label_size = tuple([int(i) for i in model_args['label_size']])
  else:
    label_size = fov_size
    model_args['label_size'] = label_size
  input_mip = FLAGS.input_mip
  overlap = [int(i) for i in FLAGS.overlap]
  num_classes = int(model_args['num_classes'])
  params = {
    'model_class': model_class,
    'model_args': model_args,
    'batch_size': FLAGS.batch_size,
    'num_classes': num_classes
  }
  
  # print('gpu', FLAGS.use_gpu)
  # if len(FLAGS.use_gpu):
  # print('rank_gpu', str(mpi_rank % FLAGS.use_gpu))

  mask_estimator = prepare_model(params, FLAGS.model_checkpoint, FLAGS.use_gpu)
  tensors_to_log = {
    "center": "center"
  }
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=1
  )

  predictions = mask_estimator.predict(
    input_fn=lambda: precomputed_utils.predict_input_fn_precomputed(
      input_volume=FLAGS.input_volume, 
      input_offset=input_offset,
      input_size=input_size,
      input_mip=input_mip,
      chunk_shape=fov_size, 
      label_shape=label_size,
      overlap=overlap,
      batch_size=FLAGS.batch_size, 
      offset=FLAGS.image_mean, 
      scale=FLAGS.image_stddev,
      var_threshold=FLAGS.var_threshold),
    predict_keys=['center', 'logits', 'class_prediction'],
    # hooks=[logging_hook],
    hooks = [],
    yield_single_examples=False
  )
  # for i, p in enumerate(predictions):
  #   logging.warning('rank %d block %d', mpi_rank, i)
  #   logging.warning('summaries: %s', p['class_prediction'].shape)
  # # output_shapes = io_utils.get_h5_shapes(FLAGS.data_volumes)
  # # output_shapes = io_utils.get_h5_shapes(FLAGS.data_volumes)
  # output_shapes = {FLAGS.output_volumes.split(':')[0]: input_size[::-1]}
  # io_utils.h5_sequential_chunk_writer_v2(
  #   predictions,
  #   output_volumes=FLAGS.output_volumes,
  #   output_shapes=output_shapes,
  #   num_classes=num_classes,
  #   chunk_shape=fov_size,
  #   label_shape=label_size,
  #   overlap=overlap,
  #   sub_bbox=FLAGS.bounding_box,
  #   mpi=FLAGS.mpi)

  _ = precomputed_utils.writer(
    predictions,
    output_volume=FLAGS.output_volume,
    output_offset=input_offset,
    output_size=input_size,
    chunk_shape=fov_size,
    label_shape=label_size,
    overlap=overlap
  )

if __name__ == '__main__':
  app.run(main)


