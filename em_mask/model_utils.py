from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
import horovod.tensorflow as hvd

tf.compat.v1.disable_eager_execution()

def ortho_cut(volume, batch_size):
  '''Concat orthogonal cuts'''
  _,z,y,x,c = volume.get_shape().as_list()
  b = batch_size
  yx = volume[0:b,z//2,:,:,:]
  zx = volume[0:b,:,y//2,:,:]
  zy = volume[0:b,:,:,x//2,:]
  logging.warn('volshape: %s', volume.shape)
  yz = tf.transpose(zy, perm=[0, 2, 1, 3])
  zz_pad = tf.zeros([b, z, z, c], dtype=tf.float32)
  logging.warn('cut: %s, %s, %s, %s', yx.shape, yz.shape, zx.shape, zz_pad.shape)
  output = tf.concat(
    [tf.concat([yx, yz], axis=2),
     tf.concat([zx, zz_pad], axis=2)],
    axis=1)
  return output

def ortho_project(volume, batch_size):
  '''Concat orthogonal cuts'''
  b,z,y,x,c = volume.get_shape().as_list()
  b = batch_size
  yx = tf.reduce_mean(volume[0:b, ...], axis=1)
  zx = tf.reduce_mean(volume[0:b, ...], axis=2)
  zy = tf.reduce_mean(volume[0:b, ...], axis=3)
  yz = tf.transpose(zy, perm=[0, 2, 1, 3])
  zz_pad = tf.zeros([b, z, z, c], dtype=tf.float32)
  output = tf.concat(
    [tf.concat([yx, yz], axis=2),
     tf.concat([zx, zz_pad], axis=2)],
    axis=1)
  return output
def ortho_project_rgb(volume, batch_size):
  b,z,y,x,c = volume.get_shape().as_list()
  zero_first = tf.zeros([b, z, y, x, 1], dtype=tf.float32)
  new_volume = tf.concat([zero_first, volume[...,1:]], axis=-1)

  color_palette = tf.constant([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,0],
    [0,1,1],
    [1,0,1]], tf.float32)
  
  rgb_volume = tf.reshape(
    tf.matmul(
      tf.reshape(new_volume, [-1, c]), color_palette[:c]),
    [b,z,y,x,3]
  )




  yx = tf.reduce_mean(rgb_volume, axis=1)
  zx = tf.reduce_mean(rgb_volume, axis=2)
  zy = tf.reduce_mean(rgb_volume, axis=3)
  yz = tf.transpose(zy, perm=[0, 2, 1, 3])
  zz_pad = tf.zeros([b, z, z, 3], dtype=tf.float32)
  output = tf.concat(
    [tf.concat([yx, yz], axis=2),
     tf.concat([zx, zz_pad], axis=2)],
    axis=1)
  return output

def mask_model_fn_legacy(features, labels, mode, params):
  model_class = params['model_class']
  model_args = params['model_args']
  batch_size = params['batch_size']
  logging.warn('at mask_model %s', features['image'].shape)
  outputs = model_class(features['image'], params['num_classes'])
  class_prediction = tf.argmax(outputs, axis=-1)
  predictions = {
    'center': features['center'],
    'logits': outputs,
    'class_prediction': class_prediction 
  }
  logging.warn('features in mask model %s, %s', features['center'], features['image'])
  if mode == tf.estimator.ModeKeys.PREDICT:
    center_op = tf.identity(features['center'], name='center')
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  
  loss = tf.losses.mean_squared_error(
    labels,
    outputs,
  )
  if mode == tf.estimator.ModeKeys.TRAIN:
          
    optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001 * hvd.size(),
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08)
    optimizer = hvd.DistributedOptimizer(optimizer, name='distributed_optimizer')


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    tf.summary.image('image', ortho_cut(features['image'], batch_size), 
      max_outputs=batch_size)
    tf.summary.image('labels', ortho_project(labels, batch_size), 
      max_outputs=batch_size)
    tf.summary.image('output', ortho_project(outputs, batch_size), 
      max_outputs=batch_size)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
  
  elif mode == tf.estimator.ModeKeys.EVAL:
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions['mask'],
                                   name='acc_op')
    metrics = {
      'accuracy': accuracy
    }
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=metrics)

def mask_model_fn_classfication(features, labels, mode, params):
  model_class = params['model_class']
  model_args = params['model_args']
  batch_size = params['batch_size']
  learning_rate = params['learning_rate']


  logits = model_class(features['image'], params['num_classes'])
  class_prediction = tf.argmax(logits, axis=-1)
  predictions = {
    'center': features['center'],
    'logits': logits,
    'class_prediction': class_prediction 
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    center_op = tf.identity(features['center'], name='center')
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  
  fov_size = model_args['fov_size']
  label_size = model_args['label_size']

  flat_logits = tf.reshape(logits, (-1, params['num_classes']), name='flat_logits')
  flat_labels = tf.reshape(labels, (-1, params['num_classes']), name='flat_labels')

  # first weight is zero, the later will evenly split 1.0
  
  loss = tf.compat.v1.losses.softmax_cross_entropy(
    onehot_labels=flat_labels,
    logits=flat_logits,
    label_smoothing=0.05
  )
  if mode == tf.estimator.ModeKeys.TRAIN:
          
    optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate * hvd.size(),
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08)
    optimizer = hvd.DistributedOptimizer(optimizer, name='distributed_optimizer')

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    tf.compat.v1.summary.image('image', ortho_cut(features['image'], batch_size), 
      max_outputs=batch_size)
    tf.compat.v1.summary.image('labels', ortho_project_rgb(labels, batch_size), 
      max_outputs=batch_size)
    tf.compat.v1.summary.image('output', ortho_project_rgb(logits, batch_size), 
      max_outputs=batch_size)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
  
  elif mode == tf.estimator.ModeKeys.EVAL:
    accuracy = tf.metrics.accuracy(labels=flat_labels,
                                   predictions=flat_logits,
                                   name='acc_op')
    metrics = {
      'accuracy': accuracy
    }
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=metrics)

def mask_model_fn_regression(features, labels, mode, params):
  '''Regression models.
  
  The logits won't be converted to onehot predictions but mean squared error wrt labels
  is computed as the loss
  '''
  model_class = params['model_class']
  model_args = params['model_args']
  batch_size = params['batch_size']

  fov_size = model_args['fov_size']

  logits = model_class(features['image'], params['num_classes'])
  if 'weights' in features:
    weights = features['weights']
    flat_weights = tf.reshape(weights, (-1, params['num_classes']), name='flat_weights')
  else:
    flat_weights = 1.0
  class_prediction = tf.squeeze(tf.greater_equal(logits, 0), axis=-1)
  predictions = {
    'center': features['center'],
    'logits': logits,
    'class_prediction': class_prediction 
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    center_op = tf.identity(features['center'], name='center')
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  
  label_size = model_args['label_size']
  learning_rate = params['learning_rate']
  flat_logits = tf.reshape(logits, (-1, params['num_classes']), name='flat_logits')
  flat_labels = tf.reshape(labels, (-1, params['num_classes']), name='flat_labels')
  logging.warning('---loss shapes %s %s', flat_labels.shape, flat_logits.shape)
  loss = tf.compat.v1.losses.mean_squared_error(
    flat_labels,
    flat_logits,
    weights=flat_weights
  )
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate * hvd.size(),
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08)
    optimizer = hvd.DistributedOptimizer(optimizer, name='distributed_optimizer')


    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    tf.compat.v1.summary.image('image', ortho_cut(features['image'], batch_size), 
      max_outputs=batch_size)
    tf.compat.v1.summary.image('labels', ortho_project(labels, batch_size), 
      max_outputs=batch_size)
    if 'weights' in features:
      tf.compat.v1.summary.image('weights', ortho_project(features['weights'], batch_size), 
        max_outputs=batch_size)

    tf.compat.v1.summary.image('output', ortho_project(logits, batch_size), 
      max_outputs=batch_size)
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
  
  elif mode == tf.estimator.ModeKeys.EVAL:
    accuracy = tf.metrics.accuracy(labels=flat_labels,
                                   predictions=flat_logits,
                                   name='acc_op')
    metrics = {
      'accuracy': accuracy
    }
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=metrics)