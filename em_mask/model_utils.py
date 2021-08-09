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
    #tf.print(tf.reduce_sum(flat_weights, axis=0))
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

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  denominator = tf.reduce_sum(y_true + y_pred)

  return 1 - numerator / denominator

def balanced_cross_entropy():
  def loss(y_true, y_pred):
    # nonlocal beta
    # if not beta:
    beta = tf.reduce_mean(1 - y_true)
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)
    
    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
    return tf.reduce_mean(o)

  return loss

def get_weight(y_true):
  positive = tf.math.greater_equal(y_true, 0.0)
  logging.warning('--pos %s', positive.shape)
  # positive = y_true > 0.0
  beta = tf.reduce_mean(1.0 - tf.cast(positive, tf.float32)) + 1e-4
  logging.warning('--beta %s', beta)
  weight = tf.where(positive, beta, 1.0 - beta)
  return weight

def mask_model_fn_binary(features, labels, mode, params):
  '''Binary classification models.
  
  The logits will be from 0.0-1.0, use dice loss with label 
  is computed as the loss
  '''
  model_class = params['model_class']
  model_args = params['model_args']
  batch_size = params['batch_size']

  # whether to use weight
  weighted = params['weighted']



  fov_size = model_args['fov_size']


  logits = model_class(features['image'], params['num_classes'])
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

  # set weight
  if 'weights' in features:
    weights = features['weights']
    flat_weights = tf.reshape(weights, (-1, params['num_classes']), name='flat_weights')
    #tf.print(tf.reduce_sum(flat_weights, axis=0))
  elif weighted:
    flat_weight = get_weight(flat_labels)
  else:
    flat_weights = 1.0

  # flat_weight = get_weight(flat_labels)
  loss = tf.compat.v1.losses.mean_squared_error(
    flat_labels,
    flat_logits,
    weights=flat_weights
  )



  # loss = tf.compat.v1.losses.mean_squared_error(
  #   flat_labels,
  #   flat_logits,
  #   weights=flat_weights
  # )
  # loss = dice_loss(
  #   flat_labels,
  #   flat_logits
  # )
  # loss = balanced_cross_entropy()(
  #   flat_labels,
  #   flat_logits
  # )
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