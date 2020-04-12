'''A 3D UNet Predicting membrane alongsie ffn. 

Each model defines a forward graph from input patches to logits
'''
import tensorflow as tf
from absl import logging
import numpy as np

def dummy_model(patches, num_classes):
  tf.print(patches)
  # inputs = tf.ones(patches.shape.as_list(), tf.float32)
  inputs = patches
  conv = tf.keras.layers.Conv3D
  # outputs = tf.layers.conv3d(inputs, 3, 3, activation='relu', padding='same', name='conv_1')
  outputs = conv(3, 3, padding='same', name='conv_1')(inputs)
  logging.warn('>>> dummy_patch_shape %s %s %s %s', 
    patches.shape, 
    tf.shape(patches),
    outputs.shape,
    tf.shape(outputs))
  return outputs
def unet_2d(patches, num_classes):
  conv = tf.layers.conv2d
  upconv = tf.layers.conv2d_transpose
  pool = tf.layers.max_pooling2d
  upsample = tf.keras.layers.UpSampling2D
  concat_axis = 3

  # _, z, y, x, c = patches.shape
  # squeezed_patches = tf.reshape(patches, [-1, y, x, c])
  logging.warn('>>>patch_shape %s %s', patches.shape, tf.shape(patches))
  # squeezed_patches = tf.squeeze(patches, axis=1)
  # squeezed_patches = tf.squeeze(patches, axis=1)
  # squeezed_patches = patches[:,0,:,:,:]
  squeezed_patches = tf.reduce_mean(patches, axis=1)
  logging.warn('squeezed %s', squeezed_patches)
  conv1 = conv(squeezed_patches, 32, 3, activation='relu', padding='same', name='conv_1a')
  conv1 = conv(conv1, 64, 3, activation='relu', padding='same', name='conv_1b')
  pool1 = pool(conv1, pool_size=2, strides=2, name='pool_1')

  conv2 = conv(pool1, 64, 3, activation='relu', padding='same', name='conv_2a')
  conv2 = conv(conv2, 128, 3, activation='relu', padding='same', name='conv_2b')
  pool2 = pool(conv2, pool_size=2, strides=2, name='pool_2')

  conv3 = conv(pool2, 128, 3, activation='relu', padding='same', name='conv_3a')
  conv3 = conv(conv3, 256, 3, activation='relu', padding='same', name='conv_3b')
  pool3 = pool(conv3, pool_size=2, strides=2, name='pool_3')

  conv4 = conv(pool3, 256, 3, activation='relu', padding='same', name='conv_4a')
  conv4 = conv(conv4, 256, 3, activation='relu', padding='same', name='conv_4b')

  upconv5 = upconv(conv4, 512, 3, strides=2, padding='same', activation=None, name='upconv_5')
  concat5 = tf.concat([conv3, upconv5], axis=3, name='concat_5')
  conv5 = conv(concat5, 256, 3, activation='relu', padding='same', name='conv_5a')
  conv5 = conv(conv5, 256, 3, activation='relu', padding='same', name='conv_5b')

  upconv6 = upconv(conv5, 256, 3, strides=2, padding='same', activation=None, name='upconv_6')
  concat6 = tf.concat([conv2, upconv6], axis=3, name='concat_6')
  conv6 = conv(concat6, 128, 3, activation='relu', padding='same', name='conv_6a')
  conv6 = conv(conv6, 128, 3, activation='relu', padding='same', name='conv_6b')

  upconv7 = upconv(conv6, 128, 3, strides=2, padding='same', activation=None, name='upconv_7')
  concat7 = tf.concat([conv1, upconv7], axis=3, name='concat_7')
  conv7 = conv(concat7, 64, 3, activation='relu', padding='same', name='conv_7a')
  conv7 = conv(conv7, 64, 3, activation='relu', padding='same', name='conv_7b')

  outputs = conv(conv7, num_classes, 3, padding='same', name='conv_7c')
  outputs = tf.expand_dims(outputs, axis=1)
  logging.warn('>>>output_shape %s', outputs.shape)


  return outputs
def unet(patches, num_classes):
  '''Simple UNet from 
  3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
  https://arxiv.org/pdf/1606.06650.pdf

  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  conv = tf.layers.conv3d
  upconv = tf.layers.conv3d_transpose
  pool = tf.layers.max_pooling3d
  upsample = tf.keras.layers.UpSampling3D
  concat_axis = 4

  conv1 = conv(patches, 32, 3, activation='relu', padding='same', name='conv_1a')
  conv1 = conv(conv1, 64, 3, activation='relu', padding='same', name='conv_1b')
  pool1 = pool(conv1, pool_size=2, strides=2, name='pool_1')

  conv2 = conv(pool1, 64, 3, activation='relu', padding='same', name='conv_2a')
  conv2 = conv(conv2, 128, 3, activation='relu', padding='same', name='conv_2b')
  pool2 = pool(conv2, pool_size=2, strides=2, name='pool_2')

  conv3 = conv(pool2, 128, 3, activation='relu', padding='same', name='conv_3a')
  conv3 = conv(conv3, 256, 3, activation='relu', padding='same', name='conv_3b')
  pool3 = pool(conv3, pool_size=2, strides=2, name='pool_3')

  conv4 = conv(pool3, 256, 3, activation='relu', padding='same', name='conv_4a')
  conv4 = conv(conv4, 256, 3, activation='relu', padding='same', name='conv_4b')

  upconv5 = upconv(conv4, 512, 3, strides=2, padding='same', activation=None, name='upconv_5')
  concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
  conv5 = conv(concat5, 256, 3, activation='relu', padding='same', name='conv_5a')
  conv5 = conv(conv5, 256, 3, activation='relu', padding='same', name='conv_5b')

  upconv6 = upconv(conv5, 256, 3, strides=2, padding='same', activation=None, name='upconv_6')
  concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
  conv6 = conv(concat6, 128, 3, activation='relu', padding='same', name='conv_6a')
  conv6 = conv(conv6, 128, 3, activation='relu', padding='same', name='conv_6b')

  upconv7 = upconv(conv6, 128, 3, strides=2, padding='same', activation=None, name='upconv_7')
  concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
  conv7 = conv(concat7, 64, 3, activation='relu', padding='same', name='conv_7a')
  conv7 = conv(conv7, 64, 3, activation='relu', padding='same', name='conv_7b')

  outputs = conv(conv7, num_classes, 3, padding='same', name='conv_7c')

  return outputs
def conv_bn_relu(inputs, filters, filter_size, name):
  logging.warn('>> inputs_shape %s %s', inputs.shape, tf.shape(inputs))
  conv = tf.layers.conv3d(inputs, filters, filter_size, activation=None, padding='same', name=name+'_conv')
  conv = tf.layers.batch_normalization(conv, name=name+'_bn')
  conv = tf.nn.relu(conv, name=name+'_relu')
  return conv

def unet_with_bn(patches, num_classes):
  '''Simple UNet from 
  3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
  https://arxiv.org/pdf/1606.06650.pdf

  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  conv = tf.layers.conv3d
  upconv = tf.layers.conv3d_transpose
  pool = tf.layers.max_pooling3d
  upsample = tf.keras.layers.UpSampling3D
  batch_norm = tf.layers.batch_normalization
  concat_axis = 4


  conv1 = conv_bn_relu(patches, 32, 3, name='conv_1a')
  conv1 = conv_bn_relu(conv1, 64, 3, name='conv_1b')
  pool1 = pool(conv1, pool_size=2, strides=2, name='pool_1')

  conv2 = conv_bn_relu(pool1, 64, 3, name='conv_2a')
  conv2 = conv_bn_relu(conv2, 128, 3, name='conv_2b')
  pool2 = pool(conv2, pool_size=2, strides=2, name='pool_2')

  conv3 = conv_bn_relu(pool2, 128, 3, name='conv_3a')
  conv3 = conv_bn_relu(conv3, 256, 3, name='conv_3b')
  pool3 = pool(conv3, pool_size=2, strides=2, name='pool_3')

  conv4 = conv_bn_relu(pool3, 256, 3, name='conv_4a')
  conv4 = conv_bn_relu(conv4, 256, 3, name='conv_4b')

  upconv5 = upconv(conv4, 512, 3, strides=2, padding='same', activation=None, name='upconv_5')
  concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
  conv5 = conv_bn_relu(concat5, 256, 3, name='conv_5a')
  conv5 = conv_bn_relu(conv5, 256, 3, name='conv_5b')

  upconv6 = upconv(conv5, 256, 3, strides=2, padding='same', activation=None, name='upconv_6')
  concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
  conv6 = conv_bn_relu(concat6, 128, 3, name='conv_6a')
  conv6 = conv_bn_relu(conv6, 128, 3, name='conv_6b')

  upconv7 = upconv(conv6, 128, 3, strides=2, padding='same', activation=None, name='upconv_7')
  concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
  conv7 = conv_bn_relu(concat7, 64, 3, name='conv_7a')
  conv7 = conv_bn_relu(conv7, 64, 3, name='conv_7b')

  outputs = conv(conv7, num_classes, 3, padding='same', name='conv_7c')

  return outputs

def crop_concat_old(a, b, name=''):
  '''Return crop concat using the smaller shape'''
  shape_a = a.shape.as_list()
  shape_b = b.shape.as_list()
  _, min_Z, min_Y, min_X, _ = [min(i, j) for i, j in zip(shape_a, shape_b)]
  # logging.warn('cutshape %s, %s, %s', min_Z, min_Y, min_X)
  return tf.concat([
    a[:, :min_Z, :min_Y, :min_X, :],
    b[:, :min_Z, :min_Y, :min_X, :]],
    axis=4,
    name=name)

def crop_concat(a, b, name=''):
  '''Return crop concat using the smaller shape, assume a larger than b'''
  shape_a = np.array(a.shape.as_list())[1:4]
  shape_b = np.array(b.shape.as_list())[1:4]
  assert (shape_a >= shape_b).all()
  assert ((shape_a - shape_b) % 2 == 0).all()
  start = (shape_a - shape_b) // 2
  end =  shape_a - start
  # _, min_Z, min_Y, min_X, _ = [min(i, j) for i, j in zip(shape_a, shape_b)]
  # logging.warn('cutshape %s, %s, %s', min_Z, min_Y, min_X)
  a_crop = a[:, start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
  return tf.concat([a_crop, b], axis=-1, name=name)

def pad_concat(a, b, name=''):
  '''Return pad concat using the larger shape, assume a larger than b'''
  shape_a = np.array(a.shape.as_list())[1:4]
  shape_b = np.array(b.shape.as_list())[1:4]
  assert (shape_a >= shape_b).all()
  assert ((shape_a - shape_b) % 2 == 0).all()
  start = (shape_a - shape_b) // 2
  end =  shape_a - start
  pad_left = (shape_a - shape_b) // 2
  pad_right = shape_a - shape_b - (shape_a - shape_b) // 2
  paddings = tf.constant(
    [[0, 0],
     [pad_left[0], pad_right[0]],
     [pad_left[1], pad_right[1]],
     [pad_left[2], pad_right[2]],
     [0, 0]])


  # _, min_Z, min_Y, min_X, _ = [min(i, j) for i, j in zip(shape_a, shape_b)]
  # logging.warn('cutshape %s, %s, %s', min_Z, min_Y, min_X)
  #a_crop = a[:, start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
  b_pad = tf.pad(b, paddings, 'SYMMETRIC')
  return tf.concat([a, b_pad], axis=-1, name=name)

def conv_bn_relu_keras(filters, filter_size, padding='same', name=None):
  def call(inputs):
    conv = tf.keras.layers.Conv3D(filters, filter_size, activation=None, padding=padding, name=name+'_conv')(inputs)
    conv = tf.keras.layers.BatchNormalization(name=name+'_bn')(conv)
    conv = tf.keras.layers.ReLU(name=name+'_relu')(conv)
    return conv
  return call
# def conv_bn_relu_keras(inputs, filters, filter_size, padding='same', name=None):
#   conv = tf.keras.layers.Conv3D(filters, filter_size, activation=None, padding=padding, name=name+'_conv')(inputs)
#   # conv = tf.keras.layers.BatchNormalization(name=name+'_bn')(conv)
#   conv = tf.keras.layers.ReLU(name=name+'_relu')(conv)
#   return conv

def conv_relu_keras(filters, filter_size, padding='same', name=None):
  def call(inputs):
    conv = tf.keras.layers.Conv3D(
      filters, 
      filter_size, 
      padding=padding, 
      # activation=tf.keras.activations.relu, 
      activation='relu', 
      kernel_initializer='he_normal',
      # kernel_regularizer=tf.keras.regularizers.l2(0.01),
      name=name+'_conv')(inputs)
    # conv = tf.keras.layers.BatchNormalization(name=name+'_bn')(conv)
    # conv = tf.keras.layers.ReLU(name=name+'_relu')(conv)
    return conv
  return call

def unet_with_bn_v2(patches, num_classes):
  '''Simple UNet from 
  3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
  https://arxiv.org/pdf/1606.06650.pdf

  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  conv = tf.layers.conv3d
  upconv = tf.layers.conv3d_transpose
  pool = tf.layers.max_pooling3d
  upsample = tf.keras.layers.UpSampling3D
  batch_norm = tf.layers.batch_normalization
  concat_axis = 4


  conv1 = conv_bn_relu(patches, 32, 3, name='conv_1a')
  conv1 = conv_bn_relu(conv1, 64, 3, name='conv_1b')
  pool1 = pool(conv1, pool_size=2, strides=2, name='pool_1')

  conv2 = conv_bn_relu(pool1, 64, 3, name='conv_2a')
  conv2 = conv_bn_relu(conv2, 128, 3, name='conv_2b')
  pool2 = pool(conv2, pool_size=2, strides=2, name='pool_2')

  conv3 = conv_bn_relu(pool2, 128, 3, name='conv_3a')
  conv3 = conv_bn_relu(conv3, 256, 3, name='conv_3b')
  pool3 = pool(conv3, pool_size=2, strides=2, name='pool_3')

  conv4 = conv_bn_relu(pool3, 256, 3, name='conv_4a')
  conv4 = conv_bn_relu(conv4, 256, 3, name='conv_4b')

  upconv5 = upconv(conv4, 512, 3, strides=2, padding='same', activation=None, name='upconv_5')
  concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
  conv5 = conv_bn_relu(concat5, 256, 3, name='conv_5a')
  conv5 = conv_bn_relu(conv5, 256, 3, name='conv_5b')

  upconv6 = upconv(conv5, 256, 3, strides=2, padding='same', activation=None, name='upconv_6')
  concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
  conv6 = conv_bn_relu(concat6, 128, 3, name='conv_6a')
  conv6 = conv_bn_relu(conv6, 128, 3, name='conv_6b')

  upconv7 = upconv(conv6, 128, 3, strides=2, padding='same', activation=None, name='upconv_7')
  concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
  conv7 = conv_bn_relu(concat7, 64, 3, name='conv_7a')
  conv7 = conv_bn_relu(conv7, 64, 3, name='conv_7b')

  outputs = conv(conv7, num_classes, 1, padding='same', name='conv_7c')

  return outputs

def unet_with_bn_noniso(patches, num_classes):
  '''Simple UNet from 
  3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
  https://arxiv.org/pdf/1606.06650.pdf

  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  conv = tf.layers.conv3d
  upconv = tf.layers.conv3d_transpose
  pool = tf.layers.max_pooling3d
  upsample = tf.keras.layers.UpSampling3D
  batch_norm = tf.layers.batch_normalization
  concat_axis = 4


  conv1 = conv_bn_relu(patches, 32, (1,3,3), name='conv_1a')
  conv1 = conv_bn_relu(conv1, 64, (1,3,3), name='conv_1b')
  pool1 = pool(conv1, pool_size=(1,2,2), strides=(1,2,2), name='pool_1')

  conv2 = conv_bn_relu(pool1, 64, (1,3,3), name='conv_2a')
  conv2 = conv_bn_relu(conv2, 128, (1,3,3), name='conv_2b')
  pool2 = pool(conv2, pool_size=(1,2,2), strides=(1,2,2), name='pool_2')

  conv3 = conv_bn_relu(pool2, 128, 3, name='conv_3a')
  conv3 = conv_bn_relu(conv3, 256, 3, name='conv_3b')
  pool3 = pool(conv3, pool_size=2, strides=2, name='pool_3')

  conv4 = conv_bn_relu(pool3, 256, 3, name='conv_4a')
  conv4 = conv_bn_relu(conv4, 256, 3, name='conv_4b')

  upconv5 = upconv(conv4, 512, 3, strides=2, padding='same', activation=None, name='upconv_5')
  concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
  conv5 = conv_bn_relu(concat5, 256, 3, name='conv_5a')
  conv5 = conv_bn_relu(conv5, 256, 3, name='conv_5b')

  upconv6 = upconv(conv5, 256, (1,3,3), strides=(1,2,2), padding='same', activation=None, name='upconv_6')
  concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
  conv6 = conv_bn_relu(concat6, 128, (1,3,3), name='conv_6a')
  conv6 = conv_bn_relu(conv6, 128, (1,3,3), name='conv_6b')

  upconv7 = upconv(conv6, 128, (1,3,3), strides=(1,2,2), padding='same', activation=None, name='upconv_7')
  concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
  conv7 = conv_bn_relu(concat7, 64, (1,3,3), name='conv_7a')
  conv7 = conv_bn_relu(conv7, 64, (1,3,3), name='conv_7b')

  outputs = conv(conv7, num_classes, 1, padding='same', name='conv_7c')

  logging.warning('conv1: %s', conv1.shape)
  logging.warning('conv2: %s', conv2.shape)
  logging.warning('conv3: %s', conv3.shape)
  logging.warning('conv4: %s', conv4.shape)
  logging.warning('conv5: %s', conv5.shape)
  logging.warning('conv6: %s', conv6.shape)
  logging.warning('conv7: %s', conv7.shape)
  logging.warning('conv8: %s', outputs.shape)

  return outputs

def unet_with_bn_noniso_v2(patches, num_classes):
  '''Simple UNet from 
  3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
  https://arxiv.org/pdf/1606.06650.pdf

  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  conv = tf.layers.conv3d
  upconv = tf.layers.conv3d_transpose
  pool = tf.layers.max_pooling3d
  upsample = tf.keras.layers.UpSampling3D
  batch_norm = tf.layers.batch_normalization
  concat_axis = 4


  conv1 = conv_bn_relu(patches, 32, (1,3,3), name='conv_1a')
  conv1 = conv_bn_relu(conv1, 64, (1,3,3), name='conv_1b')
  pool1 = pool(conv1, pool_size=(1,2,2), strides=(1,2,2), name='pool_1')

  conv2 = conv_bn_relu(pool1, 64, (1,3,3), name='conv_2a')
  conv2 = conv_bn_relu(conv2, 128, (1,3,3), name='conv_2b')
  pool2 = pool(conv2, pool_size=(1,2,2), strides=(1,2,2), name='pool_2')

  conv3 = conv_bn_relu(pool2, 128, 3, name='conv_3a')
  conv3 = conv_bn_relu(conv3, 256, 3, name='conv_3b')
  pool3 = pool(conv3, pool_size=2, strides=2, name='pool_3')

  conv4 = conv_bn_relu(pool3, 256, 3, name='conv_4a')
  conv4 = conv_bn_relu(conv4, 256, 3, name='conv_4b')

  upconv5 = upconv(conv4, 512, 2, strides=2, padding='same', activation=None, name='upconv_5')
  concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
  conv5 = conv_bn_relu(concat5, 256, 3, name='conv_5a')
  conv5 = conv_bn_relu(conv5, 256, 3, name='conv_5b')

  upconv6 = upconv(conv5, 256, (1,2,2), strides=(1,2,2), padding='same', activation=None, name='upconv_6')
  concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
  conv6 = conv_bn_relu(concat6, 128, (1,3,3), name='conv_6a')
  conv6 = conv_bn_relu(conv6, 128, (1,3,3), name='conv_6b')

  upconv7 = upconv(conv6, 128, (1,2,2), strides=(1,2,2), padding='same', activation=None, name='upconv_7')
  concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
  conv7 = conv_bn_relu(concat7, 64, (1,3,3), name='conv_7a')
  conv7 = conv_bn_relu(conv7, 64, (1,3,3), name='conv_7b')

  outputs = conv(conv7, num_classes, 1, padding='same', name='conv_7c')

  logging.warning('conv1: %s', conv1.shape)
  logging.warning('conv2: %s', conv2.shape)
  logging.warning('conv3: %s', conv3.shape)
  logging.warning('conv4: %s', conv4.shape)
  logging.warning('conv5: %s', conv5.shape)
  logging.warning('conv6: %s', conv6.shape)
  logging.warning('conv7: %s', conv7.shape)
  logging.warning('conv8: %s', outputs.shape)

  return outputs

# def unet_with_bn_noniso_keras_v2(patches, num_classes):
#   with tf.name_scope('unet'):
#     conv = tf.keras.layers.Conv3D
#     upconv = tf.keras.layers.Conv3DTranspose
#     pool = tf.keras.layers.MaxPool3D
#     batch_norm = tf.keras.layers.BatchNormalization
#     concat_axis = 4
#     padding_mode = 'same'

#     # conv0 = conv_bn_relu_keras(3, (1, 1, 1), padding='same', name='conv_0')(patches)

#     conv1 = conv_bn_relu_keras(patches, 32, (1, 3, 3), padding=padding_mode, name='conv_1a')
#     conv1 = conv_bn_relu_keras(conv1, 64, (1, 3, 3), padding=padding_mode, name='conv_1b')
#     pool1 = pool((1, 2, 2), name='pool_1')(conv1)
#     logging.warn('conv1: %s', conv1.shape)

#     conv2 = conv_bn_relu_keras(pool1, 64, (1, 3, 3), padding=padding_mode, name='conv_2a')
#     conv2 = conv_bn_relu_keras(conv2, 128, (1, 3, 3), padding=padding_mode, name='conv_2b')
#     pool2 = pool((1, 2, 2), name='pool_2')(conv2)
#     logging.warn('conv2: %s', conv2.shape)

#     conv3 = conv_bn_relu_keras(pool2, 128, (3, 3, 3), padding=padding_mode, name='conv_3a')
#     conv3 = conv_bn_relu_keras(conv3, 256, (3, 3, 3), padding=padding_mode, name='conv_3b')
#     pool3 = pool((2, 2, 2), name='pool_3')(conv3)
#     logging.warn('conv3: %s', conv3.shape)

#     conv4 = conv_bn_relu_keras(pool3, 256, (3, 3, 3), padding=padding_mode, name='conv_4a')
#     conv4 = conv_bn_relu_keras(conv4, 512, (3, 3, 3), padding=padding_mode, name='conv_4b')
#     logging.warn('conv4: %s', conv4.shape)

#     upconv5 = upconv(512, (3, 3, 3), strides=(2, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
#     concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
#     # concat5 = crop_concat(conv3, upconv5, name='concat_5')
#     conv5 = conv_bn_relu_keras(concat5, 256, (3, 3, 3), padding=padding_mode, name='conv_5a')
#     conv5 = conv_bn_relu_keras(conv5, 256, (3, 3, 3), padding=padding_mode, name='conv_5b')

#     upconv6 = upconv(256, (1, 3, 3), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
#     concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
#     # concat6 = crop_concat(conv2, upconv6, name='concat_6')
#     conv6 = conv_bn_relu_keras(concat6, 128, (1, 3, 3), padding=padding_mode, name='conv_6a')
#     conv6 = conv_bn_relu_keras(conv6, 128, (1, 3, 3), padding=padding_mode, name='conv_6b')

#     upconv7 = upconv(128, (1, 3, 3), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
#     concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
#     # concat7 = crop_concat(conv1, upconv7, name='concat_7')
#     conv7 = conv_bn_relu_keras(concat7, 64, (1, 3, 3), padding=padding_mode, name='conv_7a')
#     conv7 = conv_bn_relu_keras(conv7, 64, (1, 3, 3), padding=padding_mode, name='conv_7b')

#     output = conv_bn_relu_keras(conv7, num_classes, (1, 1, 1), padding='same', name='conv_8')

#     logging.warn('conv5: %s', conv5.shape)
#     logging.warn('conv6: %s', conv6.shape)
#     logging.warn('conv7: %s', conv7.shape)
#     logging.warn('conv8: %s', output.shape)
#     return output

def unet_with_bn_noniso_keras(patches, num_classes):
  with tf.name_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    # conv0 = conv_bn_relu_keras(3, (1, 1, 1), padding='same', name='conv_0')(patches)

    conv1 = conv_bn_relu_keras(32, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((1, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((1, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)

    conv3 = conv_bn_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((2, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_bn_relu_keras(512, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)

    upconv5 = upconv(512, (2, 2, 2), strides=(2, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
    concat5 = crop_concat(conv3, upconv5, name='concat_5')
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5b')(conv5)

    upconv6 = upconv(256, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
    concat6 = crop_concat(conv2, upconv6, name='concat_6')
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(128, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
    concat7 = crop_concat(conv1, upconv7, name='concat_7')
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    output = conv(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output

def unet_dtu_2(patches, num_classes):
  '''DTU-2 UNet from 
  Synaptic Cleft Segmentation in Non-isotropic Volume Electron Microscopy of the 
  Complete Drosophila Brain
  https://link.springer.com/content/pdf/10.1007%2F978-3-030-00934-2_36.pdf
  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  with tf.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'


    # conv1 = conv(12, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    # conv1 = conv(12, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    # pool1 = pool((1, 3, 3), name='pool_1')(conv1)

    # conv2 = conv(72, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    # conv2 = conv(72, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    # pool2 = pool((1, 3, 3), name='pool_2')(conv2)

    # conv3 = conv(432, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    # conv3 = conv(432, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    # pool3 = pool((3, 3, 3), name='pool_3')(conv3)

    # conv4 = conv(2592, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    # conv4 = conv(2592, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)

    with tf.variable_scope('block_1'):
      conv1 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
      conv1 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
      pool1 = pool((1, 3, 3), name='pool_1')(conv1)

    with tf.variable_scope('block_2'):
      conv2 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
      conv2 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
      pool2 = pool((1, 3, 3), name='pool_2')(conv2)

    with tf.variable_scope('block_3'):
      conv3 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
      conv3 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
      pool3 = pool((3, 3, 3), name='pool_3')(conv3)

    with tf.variable_scope('block_4'):
      conv4 = conv_relu_keras(2592, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
      conv4 = conv_relu_keras(2592, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv1: %s', conv1.shape)
    logging.warn('conv2: %s', conv2.shape)
    logging.warn('conv3: %s', conv3.shape)
    logging.warn('conv4: %s', conv4.shape)

    with tf.variable_scope('block_5'):
      upconv5 = upconv(432, (3, 3, 3), strides=(3, 3, 3), activation='relu', padding=padding_mode, name='upconv_5')(conv4)
      # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
      concat5 = crop_concat(conv3, upconv5, name='concat_5')
      conv5 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_5')(concat5)

    with tf.variable_scope('block_6'):
      upconv6 = upconv(72, (1, 3, 3), strides=(1, 3, 3), activation='relu', padding=padding_mode, name='upconv_6')(conv5)
      # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
      concat6 = crop_concat(conv2, upconv6, name='concat_6')
      conv6 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_6')(concat6)

    with tf.variable_scope('block_7'):
      upconv7 = upconv(12, (1, 3, 3), strides=(1, 3, 3), activation='relu', padding=padding_mode, name='upconv_7')(conv6)
      # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
      concat7 = crop_concat(conv1, upconv7, name='concat_7')
      conv7 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_7')(concat7)

      # output = conv_relu_keras(num_classes, (1, 1, 1), padding=padding_mode, name='conv_8')(conv7)
      output = conv(num_classes, (1, 1, 1), padding=padding_mode, name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)

    return output

def unet_dtu_2_pad_concat(patches, num_classes):
  '''DTU-2 UNet from 
  Synaptic Cleft Segmentation in Non-isotropic Volume Electron Microscopy of the 
  Complete Drosophila Brain
  https://link.springer.com/content/pdf/10.1007%2F978-3-030-00934-2_36.pdf
  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'


    # conv1 = conv(12, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    # conv1 = conv(12, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    # pool1 = pool((1, 3, 3), name='pool_1')(conv1)

    # conv2 = conv(72, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    # conv2 = conv(72, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    # pool2 = pool((1, 3, 3), name='pool_2')(conv2)

    # conv3 = conv(432, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    # conv3 = conv(432, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    # pool3 = pool((3, 3, 3), name='pool_3')(conv3)

    # conv4 = conv(2592, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    # conv4 = conv(2592, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)

    with tf.compat.v1.variable_scope('block_1'):
      conv1 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
      conv1 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
      pool1 = pool((1, 3, 3), name='pool_1')(conv1)

    with tf.compat.v1.variable_scope('block_2'):
      conv2 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
      conv2 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
      pool2 = pool((1, 3, 3), name='pool_2')(conv2)

    with tf.compat.v1.variable_scope('block_3'):
      conv3 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
      conv3 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
      pool3 = pool((3, 3, 3), name='pool_3')(conv3)

    with tf.compat.v1.variable_scope('block_4'):
      conv4 = conv_relu_keras(2592, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
      conv4 = conv_relu_keras(2592, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv1: %s', conv1.shape)
    logging.warn('conv2: %s', conv2.shape)
    logging.warn('conv3: %s', conv3.shape)
    logging.warn('conv4: %s', conv4.shape)

    with tf.compat.v1.variable_scope('block_5'):
      upconv5 = upconv(432, (3, 3, 3), strides=(3, 3, 3), activation='relu', padding=padding_mode, name='upconv_5')(conv4)
      # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
      concat5 = pad_concat(conv3, upconv5, name='concat_5')
      conv5 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_5')(concat5)

    with tf.compat.v1.variable_scope('block_6'):
      upconv6 = upconv(72, (1, 3, 3), strides=(1, 3, 3), activation='relu', padding=padding_mode, name='upconv_6')(conv5)
      # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
      concat6 = pad_concat(conv2, upconv6, name='concat_6')
      conv6 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_6')(concat6)

    with tf.compat.v1.variable_scope('block_7'):
      upconv7 = upconv(12, (1, 3, 3), strides=(1, 3, 3), activation='relu', padding=padding_mode, name='upconv_7')(conv6)
      # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
      concat7 = pad_concat(conv1, upconv7, name='concat_7')
      conv7 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_7')(concat7)

      # output = conv_relu_keras(num_classes, (1, 1, 1), padding=padding_mode, name='conv_8')(conv7)
      output = conv(num_classes, (1, 1, 1), padding=padding_mode, name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)

    return output

def unet_dtu_2_pad_concat_v2(patches, num_classes):
  '''DTU-2 UNet from 
  Synaptic Cleft Segmentation in Non-isotropic Volume Electron Microscopy of the 
  Complete Drosophila Brain
  https://link.springer.com/content/pdf/10.1007%2F978-3-030-00934-2_36.pdf
  no crop or pad, assuming patch shape has dimensions of powers of 2
  '''
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    with tf.compat.v1.variable_scope('block_1'):
      conv1 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
      conv1 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
      pool1 = pool((1, 3, 3), name='pool_1')(conv1)

    with tf.compat.v1.variable_scope('block_2'):
      conv2 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
      conv2 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
      pool2 = pool((1, 3, 3), name='pool_2')(conv2)

    with tf.compat.v1.variable_scope('block_3'):
      conv3 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
      conv3 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
      pool3 = pool((3, 3, 3), name='pool_3')(conv3)

    with tf.compat.v1.variable_scope('block_4'):
      conv4 = conv_relu_keras(2592, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
      conv4 = conv_relu_keras(2592, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv1: %s', conv1.shape)
    logging.warn('conv2: %s', conv2.shape)
    logging.warn('conv3: %s', conv3.shape)
    logging.warn('conv4: %s', conv4.shape)

    with tf.compat.v1.variable_scope('block_5'):
      upconv5 = upconv(432, (3, 3, 3), strides=(3, 3, 3), activation='relu', padding=padding_mode, name='upconv_5')(conv4)
      # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
      logging.warn('upconv5: %s', upconv5.shape)
      concat5 = pad_concat(conv3, upconv5, name='concat_5')
      conv5 = conv_relu_keras(432, (3, 3, 3), padding=padding_mode, name='conv_5')(concat5)

    with tf.compat.v1.variable_scope('block_6'):
      upconv6 = upconv(72, (1, 3, 3), strides=(1, 3, 3), activation='relu', padding=padding_mode, name='upconv_6')(conv5)
      # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
      concat6 = pad_concat(conv2, upconv6, name='concat_6')
      conv6 = conv_relu_keras(72, (1, 3, 3), padding=padding_mode, name='conv_6')(concat6)

    with tf.compat.v1.variable_scope('block_7'):
      upconv7 = upconv(12, (1, 3, 3), strides=(1, 3, 3), activation='relu', padding=padding_mode, name='upconv_7')(conv6)
      # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
      concat7 = pad_concat(conv1, upconv7, name='concat_7')
      conv7 = conv_relu_keras(12, (1, 3, 3), padding=padding_mode, name='conv_7')(concat7)

      # output = conv_relu_keras(num_classes, (1, 1, 1), padding=padding_mode, name='conv_8')(conv7)
      output = conv(num_classes, (1, 1, 1), padding=padding_mode, name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)

    return output
def conv_pool_model(patches, num_classes):
  conv = tf.keras.layers.Conv3D
  # upconv = tf.keras.layers.Conv3DTranpose
  pool = tf.keras.layers.MaxPool3D
  batch_norm = tf.keras.layers.BatchNormalization

  conv1 = conv(32, 3, name='conv_1', padding='valid')(patches)
  pool1 = pool((1,2,2), name='pool_1')(conv1)
  pool1 = batch_norm(axis=-1)(pool1)

  conv2 = conv(32, 3, name='conv_2', padding='valid')(pool1)
  pool2 = pool((1,2,2), name='pool_2')(conv2)
  pool2 = batch_norm(axis=-1)(pool2)

  conv3 = conv(32, 3, name='conv_3', padding='valid')(pool2)
  pool3 = pool((2,2,2), name='pool_2')(conv3)
  pool3 = batch_norm(axis=-1)(pool3)

  conv4 = conv(512, 4, name='conv_5', padding='valid')(pool3)
  outputs = conv(num_classes, 1, name='conv_6', padding='valid')(conv4)
  print('>>>', outputs.shape)
  return outputs


def shallow_z_unet(patches, num_classes):
  """Unet for 3D patches with small z dimension.

  Args:
    patches: (b, z, y, x, c) inputs
    num_classes: output channels
  Returns:
    output tensor
  """
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_bn_relu_keras(32, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((1, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((1, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)


    conv3 = conv_bn_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((1, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_bn_relu_keras(512, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)

    upconv5 = upconv(512, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
    concat5 = crop_concat(conv3, upconv5, name='concat_5')
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5b')(conv5)

    upconv6 = upconv(256, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
    concat6 = crop_concat(conv2, upconv6, name='concat_6')
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(128, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
    concat7 = crop_concat(conv1, upconv7, name='concat_7')
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    # conv8 = conv()
    conv8 = conv(num_classes, (1, 1, 1), padding='valid', name='conv_8')(conv7)
    output = tf.math.reduce_mean(conv8, axis=1, keepdims=True, name='mean9')

    # output = conv(num_classes, (5, 1, 1), padding='valid', name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output

def shallow_z_unet_v2(patches, num_classes):
  """Unet for 3D patches with small z dimension.

  Args:
    patches: (b, z, y, x, c) inputs
    num_classes: output channels
  Returns:
    output tensor
  """
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_bn_relu_keras(32, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((1, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((1, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)


    conv3 = conv_bn_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((5, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_bn_relu_keras(512, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)

    upconv5 = upconv(512, (1, 2, 2), strides=(5, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
    concat5 = crop_concat(conv3, upconv5, name='concat_5')
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5b')(conv5)

    upconv6 = upconv(256, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
    concat6 = crop_concat(conv2, upconv6, name='concat_6')
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(128, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
    concat7 = crop_concat(conv1, upconv7, name='concat_7')
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    # conv8 = conv()
    conv8 = conv(num_classes, (1, 1, 1), padding='valid', name='conv_8')(conv7)
    output = tf.math.reduce_mean(conv8, axis=1, keepdims=True, name='mean9')

    # output = conv(num_classes, (5, 1, 1), padding='valid', name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output

def shallow_z_unet_v3(patches, num_classes):
  """Unet for 3D patches with small z dimension.

  Args:
    patches: (b, z, y, x, c) inputs
    num_classes: output channels
  Returns:
    output tensor
  """
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_bn_relu_keras(32, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((1, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((1, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)


    conv3 = conv_bn_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((5, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_bn_relu_keras(512, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)

    upconv5 = upconv(512, (1, 2, 2), strides=(5, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
    concat5 = crop_concat(conv3, upconv5, name='concat_5')
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_bn_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5b')(conv5)

    upconv6 = upconv(256, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
    concat6 = crop_concat(conv2, upconv6, name='concat_6')
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_bn_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(128, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
    concat7 = crop_concat(conv1, upconv7, name='concat_7')
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_bn_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    # conv8 = conv()
    conv8 = conv(num_classes, (1, 1, 1), padding='valid', name='conv_8')(conv7)
    output = tf.math.reduce_mean(conv8, axis=1, keepdims=True, name='mean9')

    # output = conv(num_classes, (5, 1, 1), padding='valid', name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output

def shallow_z_unet_v4(patches, num_classes):
  """Unet for 3D patches with small z dimension.

  Args:
    patches: (b, z, y, x, c) inputs
    num_classes: output channels
  Returns:
    output tensor
  """
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_relu_keras(32, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((1, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((1, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)


    conv3 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((1, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_relu_keras(512, (1, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)

    upconv5 = upconv(512, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
    concat5 = crop_concat(conv3, upconv5, name='concat_5')
    conv5 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_5b')(conv5)

    upconv6 = upconv(256, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
    concat6 = crop_concat(conv2, upconv6, name='concat_6')
    conv6 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(128, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
    concat7 = crop_concat(conv1, upconv7, name='concat_7')
    conv7 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    # conv8 = conv()
    conv8 = conv(num_classes, (1, 1, 1), padding='valid', name='conv_8')(conv7)
    output = tf.math.reduce_mean(conv8, axis=1, keepdims=True, name='mean9')

    # output = conv(num_classes, (5, 1, 1), padding='valid', name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output

def shallow_z_unet_v5(patches, num_classes):
  """Unet for 3D patches with small z dimension.

  Args:
    patches: (b, z, y, x, c) inputs
    num_classes: output channels
  Returns:
    output tensor
  """
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_relu_keras(32, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((1, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((1, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)


    conv3 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((1, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_relu_keras(512, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)

    upconv5 = upconv(512, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    # concat5 = tf.concat([conv3, upconv5], axis=4, name='concat_5')
    concat5 = crop_concat(conv3, upconv5, name='concat_5')
    conv5 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5b')(conv5)

    upconv6 = upconv(256, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    # concat6 = tf.concat([conv2, upconv6], axis=4, name='concat_6')
    concat6 = crop_concat(conv2, upconv6, name='concat_6')
    conv6 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(128, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    # concat7 = tf.concat([conv1, upconv7], axis=4, name='concat_7')
    concat7 = crop_concat(conv1, upconv7, name='concat_7')
    conv7 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    # conv8 = conv()
    conv8 = conv(num_classes, (1, 1, 1), padding='valid', name='conv_8')(conv7)
    output = conv8[:, 2:3, :, :, :]
    # output = tf.math.reduce_mean(conv8, axis=1, keepdims=True, name='mean9')

    # output = conv(num_classes, (5, 1, 1), padding='valid', name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output

def tissue_model(patches, num_classes):
  with tf.compat.v1.variable_scope('tissue_model'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'valid'

    conv1 = conv(64, (1, 3, 3), padding=padding_mode, name='conv_1')(patches)
    pool1 = pool((1, 2, 2), strides=(1, 2, 2), name='pool_1')(conv1)

    conv2 = conv(64, (1, 3, 3), padding=padding_mode, name='conv_2')(pool1)
    pool2 = pool((1, 2, 2), strides=(1, 2, 2), name='pool_2')(conv2)

    conv3 = conv(64, (1, 3, 3), padding=padding_mode, name='conv_3')(pool2)
    pool3 = pool((1, 2, 2), strides=(1, 2, 2), name='pool_3')(conv3)

    conv4 = conv(16, (1, 3, 3), padding=padding_mode, name='conv_4')(pool3)

    conv5 = conv(512, (1, 1, 1), padding=padding_mode, name='conv_5')(conv4)

    output = conv(num_classes, (1, 1, 1), padding='valid', name='conv_6')(conv5)

    logging.warn('conv1: %s', conv1.shape)
    logging.warn('pool1: %s', pool1.shape)
    logging.warn('conv2: %s', conv2.shape)
    logging.warn('conv3: %s', conv3.shape)
    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', output.shape)


    # output = tf.math.reduce_mean(conv8, axis=1, keepdims=True, name='mean9')

    # output = conv(num_classes, (5, 1, 1), padding='valid', name='conv_8')(conv7)

    return output
  
def pseudo_2d_unet(patches, num_classes):
  with tf.compat.v1.variable_scope('unet'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((1, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((1, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)


    conv3 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((1, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_relu_keras(512, (1, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_relu_keras(512, (1, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    pool4 = pool((1, 2, 2), name='pool_4')(conv4)
    logging.warn('conv4: %s', conv4.shape)

    conv5 = conv_relu_keras(1024, (1, 3, 3), padding=padding_mode, name='conv_5a')(pool4)
    conv5 = conv_relu_keras(1024, (1, 3, 3), padding=padding_mode, name='conv_5b')(conv5)
    logging.warn('conv5: %s', conv5.shape)


    upconv6 = upconv(512, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    concat6 = crop_concat(conv4, upconv6, name='concat_6')
    conv6 = conv_relu_keras(512, (1, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_relu_keras(512, (1, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(256, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    concat7 = crop_concat(conv3, upconv7, name='concat_7')
    conv7 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_relu_keras(256, (1, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    upconv8 = upconv(128, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_8')(conv7)
    concat8 = crop_concat(conv2, upconv8, name='concat_8')
    conv8 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_8a')(concat8)
    conv8 = conv_relu_keras(128, (1, 3, 3), padding=padding_mode, name='conv_8b')(conv8)

    upconv9 = upconv(64, (1, 2, 2), strides=(1, 2, 2), padding=padding_mode, name='upconv_9')(conv8)
    concat9 = crop_concat(conv1, upconv9, name='concat_9')
    conv9 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_9a')(concat9)
    conv9 = conv_relu_keras(64, (1, 3, 3), padding=padding_mode, name='conv_9b')(conv9)
    conv9 = conv_relu_keras(2, (1, 3, 3), padding='same', name='conv_9c')(conv9)
    output = conv(num_classes, (1, 1, 1), padding='same', name='conv_9d')(conv9)
    # output = tf.math.reduce_mean(conv8, axis=1, keepdims=True, name='mean9')
    # output = conv(num_classes, (5, 1, 1), padding='valid', name='conv_8')(conv7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', conv8.shape)
    logging.warn('conv9: %s', conv9.shape)
    logging.warn('output: %s', output.shape)
    return output
