import tensorflow as tf
from tensorflow.keras.layers import (
  Conv3D,
  Conv3DTranspose,
  BatchNormalization,
  Add,
  Multiply,
  Activation
)
import logging

from .unets import conv_relu_keras, crop_concat



def attention_block(x, shortcut, i_filters):
  g1 = Conv3D(i_filters,kernel_size=1)(shortcut) 
  # g1 = BatchNormalization()(g1)

  x1 = Conv3D(i_filters,kernel_size=1)(x) 
  # x1 = BatchNormalization()(x1)

  g1_x1 = Add()([g1,x1])
  psi = Activation('relu')(g1_x1)
  psi = Conv3D(1,kernel_size=1)(psi) 
  # psi = BatchNormalization()(psi)
  psi = Activation('sigmoid')(psi)
  x = Multiply()([x,psi])
  return x

def attention_unet_model(patches, num_classes):
  with tf.compat.v1.variable_scope('attention'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_relu_keras(32, (3, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((2, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((2, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)

    conv3 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((2, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_relu_keras(512, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)


    upconv5 = upconv(512, (2, 2, 2), strides=(2, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    attn_5 = attention_block(upconv5, conv3, 512)
    # concat5 = crop_concat(conv3, upconv5, name='concat_5')
    concat5 = crop_concat(attn_5, upconv5, name='concat_5')
    conv5 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5b')(conv5)

    upconv6 = upconv(256, (2, 2, 2), strides=(2, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    attn_6 = attention_block(upconv6, conv2, 256)
    #  = crop_concat(conv2, upconv6, name='concat_6')
    concat6 = crop_concat(attn_6, upconv6, name='concat_6')
    conv6 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_6b')(conv6)

    upconv7 = upconv(128, (2, 2, 2), strides=(2, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    attn_7 = attention_block(upconv7, conv1, 128)
    # concat7 = crop_concat(conv1, upconv7, name='concat_7')
    concat7 = crop_concat(attn_7, upconv7, name='concat_7')
    conv7 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_7b')(conv7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    output = conv(num_classes, (1, 1, 1), padding='same', name='conv_8')(attn_7)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output

def attention_detection_model(patches, num_classes):
  with tf.compat.v1.variable_scope('attention'):
    conv = tf.keras.layers.Conv3D
    upconv = tf.keras.layers.Conv3DTranspose
    pool = tf.keras.layers.MaxPool3D
    batch_norm = tf.keras.layers.BatchNormalization
    concat_axis = 4
    padding_mode = 'same'

    conv1 = conv_relu_keras(32, (3, 3, 3), padding=padding_mode, name='conv_1a')(patches)
    conv1 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_1b')(conv1)
    pool1 = pool((2, 2, 2), name='pool_1')(conv1)
    logging.warn('conv1: %s', conv1.shape)

    conv2 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_2a')(pool1)
    conv2 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_2b')(conv2)
    pool2 = pool((2, 2, 2), name='pool_2')(conv2)
    logging.warn('conv2: %s', conv2.shape)

    conv3 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_3a')(pool2)
    conv3 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_3b')(conv3)
    pool3 = pool((2, 2, 2), name='pool_3')(conv3)
    logging.warn('conv3: %s', conv3.shape)

    conv4 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_4a')(pool3)
    conv4 = conv_relu_keras(512, (3, 3, 3), padding=padding_mode, name='conv_4b')(conv4)
    logging.warn('conv4: %s', conv4.shape)


    upconv5 = upconv(512, (2, 2, 2), strides=(2, 2, 2), padding=padding_mode, name='upconv_5')(conv4)
    attn5 = attention_block(upconv5, conv3, 512)
    # concat5 = crop_concat(conv3, upconv5, name='concat_5')
    # concat5 = crop_concat(attn_5, upconv5, name='concat_5')
    # conv5 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5a')(concat5)
    conv5 = conv_relu_keras(256, (3, 3, 3), padding=padding_mode, name='conv_5b')(attn5)

    upconv6 = upconv(256, (2, 2, 2), strides=(2, 2, 2), padding=padding_mode, name='upconv_6')(conv5)
    attn6 = attention_block(upconv6, conv2, 256)
    #  = crop_concat(conv2, upconv6, name='concat_6')
    # concat6 = crop_concat(attn_6, upconv6, name='concat_6')
    # conv6 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_6a')(concat6)
    conv6 = conv_relu_keras(128, (3, 3, 3), padding=padding_mode, name='conv_6b')(attn6)

    upconv7 = upconv(128, (2, 2, 2), strides=(2, 2, 2), padding=padding_mode, name='upconv_7')(conv6)
    attn7 = attention_block(upconv7, conv1, 128)
    # concat7 = crop_concat(conv1, upconv7, name='concat_7')
    # concat7 = crop_concat(attn_7, upconv7, name='concat_7')
    # conv7 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_7a')(concat7)
    conv7 = conv_relu_keras(64, (3, 3, 3), padding=padding_mode, name='conv_7b')(attn7)

    # output = conv_bn_relu_keras(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    output = conv(num_classes, (1, 1, 1), padding='same', name='conv_8')(conv7)
    # logging.warn('attn5: %s', attn_5.shape)
    # logging.warn('attn6: %s', attn_6.shape)
    # logging.warn('attn7: %s', attn_7.shape)

    logging.warn('conv5: %s', conv5.shape)
    logging.warn('conv6: %s', conv6.shape)
    logging.warn('conv7: %s', conv7.shape)
    logging.warn('conv8: %s', output.shape)
    return output