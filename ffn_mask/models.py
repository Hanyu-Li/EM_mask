'''A 3D UNet Predicting membrane alongsie ffn. 

Each model defines a forward graph from input patches to logits
'''
import tensorflow as tf
from absl import logging
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
