'''Merge hotspot detection model training.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import cloudvolume
# import tensorflow as tf

from ffn_mask.merge_utils import *
tf.compat.v1.disable_eager_execution()

import horovod.tensorflow as hvd
from ffn.training.import_util import import_symbol
from ffn_mask import io_utils
from ffn_mask import model_utils

import json

FLAGS = flags.FLAGS

flags.DEFINE_string('segmentation_vol', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing uint8 '
                    'image data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_integer('mip', 0, '')
# flags.DEFINE_string('chunk_size', '128,128,128', '')
flags.DEFINE_string('factor', '4,4,1', '')
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
flags.DEFINE_integer('max_steps', 100000, '')
flags.DEFINE_boolean('rotation', False, '')

def main(unused_argv):
  # chunk_size = np.array([int(i) for i in FLAGS.chunk_size.split(',')])
  model_args = json.loads(FLAGS.model_args)
  chunk_size = tuple([int(i) for i in model_args['fov_size']])
  factor = np.array([int(i) for i in FLAGS.factor.split(',')])
  seg_cv = cloudvolume.CloudVolume('file://%s' % FLAGS.segmentation_vol, 
    mip=FLAGS.mip, progress=False)
  resolution = seg_cv.meta.resolution(FLAGS.mip)

  # sample_loc = np.array([9553, 13099, 187])
  # sample_loc = sample_loc // factor
  # sample_bb = loc_to_bbox(sample_loc, chunk_size)
  # sample_chunk = np.array(seg_cv[sample_bb])[..., 0]
  # offset = sample_loc - chunk_size // 2

  # pair_dicts = training_data_prep_v2(
  #   sample_chunk,
  #   resolution=resolution, 
  #   offset=offset, 
  #   rescale=factor)
  # in_out = build_training_data(sample_chunk, pair_dicts)
  # print(len(in_out))

  # ds = merge_input_fn(seg_cv, chunk_size, factor, 
  #   n_samples=200, batch_size=FLAGS.batch_size)
  # # value = tf.compat.v1.data.make_initializable_iterator(ds).get_next()
  # value = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
  # with tf.compat.v1.Session() as sess:
  #   for _ in range(40):
  #     features, label = sess.run(value)
  #     print(features['center'], features['id_a'], features['id_b'])
  #     print(features['fuse_mask'].shape, label.shape)

  # Training
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
  save_checkpoints_secs = 180 if hvd.rank() == 0 else None

  config=tf.estimator.RunConfig( 
    model_dir=model_dir,
    save_summary_steps=save_summary_steps,
    save_checkpoints_secs=save_checkpoints_secs,
    session_config=sess_config,
    keep_checkpoint_max=100,
  )
  mask_estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=config,
    params=params
    )
  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
  # logging_hook = tf.estimator.LoggingTensorHook(
  #   {'flat_logits': 'flat_logits',
  #    'flat_labels': 'flat_labels'},
  #    every_n_iter=100,
  # )
  input_fn = lambda: merge_input_fn(
    seg_cv, 
    chunk_size, 
    factor, 
    n_samples=FLAGS.max_steps, 
    batch_size=FLAGS.batch_size)

  mask_estimator.train(
    input_fn=input_fn,
    steps=FLAGS.max_steps,
    hooks=[bcast_hook])#, logging_hook])

if __name__ == '__main__':
  app.run(main)


