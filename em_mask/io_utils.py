from packaging import version
import tensorflow as tf
if version.parse(tf.__version__) < version.parse('2.0.0'):
  tfa = tf.contrib
else:
  import tensorflow_addons as tfa
import numpy as np
import h5py
import itertools
from skimage.segmentation import find_boundaries
from ffn.utils import bounding_box
from ffn.training import inputs
from ffn.training.import_util import import_symbol
from absl import logging

from google.protobuf import text_format
from ffn.utils import bounding_box_pb2
from ffn.utils import bounding_box
from ffn.utils import geom_utils
from ffn.training.mask import crop

import horovod.tensorflow as hvd
import mpi4py
from mpi4py import MPI
import cloudvolume
tf.compat.v1.disable_eager_execution()
comm = MPI.COMM_WORLD
rank = comm.rank

def labels_to_membrane(labels):
  return np.logical_or(np.asarray(labels)==0, 
                       find_boundaries(labels, mode='thick'))
def _load_from_numpylike_v2(coord, volume, start_offset, chunk_shape, axes='zyx'):
  starts = np.array(coord) - start_offset
  slc = bounding_box.BoundingBox(start=starts, size=chunk_shape).to_slice()
  data = volume[slc[2], slc[1], slc[0], :]
  return data


def _load_from_numpylike_with_pad(coord, volume, pad_start, pad_end, chunk_shape,
                                  sample_start=None, sample_size=None):
  '''load from numpy like with padding, all in zyx order.'''
  real_volume_shape = volume.shape
  pad_bbox = bounding_box.BoundingBox(start=(0,0,0), size=pad_start+real_volume_shape+pad_end)
  if sample_start is None and sample_size is None:
    real_bbox = bounding_box.BoundingBox(start=pad_start, 
      size=real_volume_shape)
  else:
    real_bbox = bounding_box.BoundingBox(start=pad_start+sample_start, size=sample_size)
  tentative_bbox = bounding_box.BoundingBox(start=coord-chunk_shape//2, size=chunk_shape)
  actual_bbox = bounding_box.intersection(tentative_bbox, real_bbox)
  if not actual_bbox:
    return None
  read_bbox = actual_bbox.adjusted_by(start=-pad_start, end=-pad_start)
  write_bbox = bounding_box.BoundingBox(start=actual_bbox.start-tentative_bbox.start, size=actual_bbox.size)
  output = np.zeros(chunk_shape, dtype=np.uint8)
  w_slc = write_bbox.to_slice()
  w_slc = tuple([w_slc[i] for i in [2,1,0]])
  r_slc = read_bbox.to_slice()
  r_slc = tuple([r_slc[i] for i in [2,1,0]])
  
  
  output[w_slc] = volume[r_slc]
  output = np.expand_dims(output, 4)
  return output

def h5_random_chunk_generator(data_volumes, label_volumes, num_classes, chunk_shape=(32, 64, 64)):
  '''Randomly generate chunks from volumes'''
  image_volume_map = {}
  for vol in data_volumes.split(','):
    volname, path, dataset = vol.split(':')
    image_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)

  label_volume_map = {}
  for vol in label_volumes.split(','):
    volname, path, dataset = vol.split(':')
    label_volume_map[volname] = tf.keras.utils.to_categorical(
      h5py.File(path, 'r')[dataset])
  def gen():
    chunk_offset = (np.array(chunk_shape)) // 2
    for key, val in image_volume_map.items():
      data_shape = val.shape
      valid_max = [d-c for d,c in zip(data_shape, chunk_shape)]
      for i in itertools.count(1):
        center = np.floor(np.random.rand(3) * valid_max).astype(np.int64) + chunk_offset
        image, label = (
          _load_from_numpylike_v2(center, val, chunk_offset, chunk_shape),
          _load_from_numpylike_v2(center, label_volume_map[key], chunk_offset, chunk_shape))
        yield (center, image, label)
  return gen

def parser(proto):
  examples = tf.io.parse_single_example(proto, features=dict(
      center=tf.io.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
      label_volume_name=tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
  ))
  coord = examples['center']
  volname = examples['label_volume_name']
  return coord, volname

def h5_sequential_chunk_generator(data_volumes, 
                                  chunk_shape=(32, 64, 64), 
                                  overlap=(0, 0, 0),
                                  bounding_box=None,
                                  var_threshold=10):
  '''Sequentially generate chunks(with overlap) from volumes'''
  image_volume_map = {}
  for vol in data_volumes.split(','):
    volname, path, dataset = vol.split(':')
    image_volume_map[volname] = h5py.File(path,'r')[dataset]

  chunk_shape = np.array(chunk_shape)
  chunk_offset = chunk_shape // 2
  overlap = np.array(overlap)
  
  if bounding_box:
    sample_bbox = bounding_box_pb2.BoundingBox()
    text_format.Parse(bounding_box, sample_bbox)
    sample_start = geom_utils.ToNumpy3Vector(sample_bbox.start)
    sample_size = geom_utils.ToNumpy3Vector(sample_bbox.size)
    print(sample_start, sample_size)
  else:
    sample_start = None
    sample_size = None
  
  def gen():
    for key, val in image_volume_map.items():
      data_shape = np.array(val.shape)
      step_shape = chunk_shape - overlap
      step_counts = (data_shape - 1) // step_shape + 1
      pad_start = overlap // 2
      # pad zeros at end to ensure modular zero
      pad_end = step_counts * step_shape + pad_start - data_shape
      grid_zyx = [np.arange(j)*i+d//2 for i,j,d in zip(step_shape, step_counts, chunk_shape)]
      grid = np.array(np.meshgrid(*grid_zyx)).T.reshape(-1, 3)
      
      for i in range(grid.shape[0]):
        center = grid[i]
        image = _load_from_numpylike_with_pad(center, val, pad_start, pad_end, 
          chunk_shape, sample_start, sample_size )

        if image is not None:
          if np.var(image[...]) > var_threshold:
            yield (center, image)
          else:
            logging.info('skipped chunk %s', str(center))
  return gen
def h5_sequential_chunk_generator_v2(data_volumes,
                                     chunk_shape=(64, 64, 32),
                                     overlap=(0, 0, 0),
                                     bounding_box=None,
                                     var_threshold=10, 
                                     data_axes='zyx'):
  '''Sequentially generate chunks(with overlap) from volumes'''
  image_volume_map = {}
  for vol in data_volumes.split(','):
    volname, path, dataset = vol.split(':')
    image_volume_map[volname] = h5py.File(path,'r')[dataset]

  chunk_shape = np.array(chunk_shape)
  chunk_offset = chunk_shape // 2
  overlap = np.array(overlap)
  if data_axes == 'zyx':
    chunk_shape = chunk_shape[::-1]
    overlap = overlap[::-1]
  
  if bounding_box:
    sample_bbox = bounding_box_pb2.BoundingBox()
    text_format.Parse(bounding_box, sample_bbox)
    sample_start = geom_utils.ToNumpy3Vector(sample_bbox.start)
    sample_size = geom_utils.ToNumpy3Vector(sample_bbox.size)
    print(sample_start, sample_size)
  else:
    sample_start = None
    sample_size = None
  
  def gen():
    for key, val in image_volume_map.items():
      data_shape = np.array(val.shape)
      step_shape = chunk_shape - overlap
      step_counts = (data_shape - 1) // step_shape + 1
      pad_start = overlap // 2
      # pad zeros at end to ensure modular zero
      pad_end = step_counts * step_shape + pad_start - data_shape
      grid_zyx = [np.arange(j)*i+d//2 for i,j,d in zip(step_shape, step_counts, chunk_shape)]
      grid = np.array(np.meshgrid(*grid_zyx)).T.reshape(-1, 3)
      
      for i in range(grid.shape[0]):
        center = grid[i]
        image = _load_from_numpylike_with_pad(center, val, pad_start, pad_end, 
          chunk_shape, sample_start, sample_size )

        if image is not None:
          if np.var(image[...]) > var_threshold:
            yield (center, image)
          else:
            logging.info('skipped chunk %s', str(center))
  return gen

def get_h5_shapes(data_volumes):
  image_volume_shape_map = {}
  for vol in data_volumes.split(','):
    volname, path, dataset = vol.split(':')
    image_volume_shape_map[volname] = h5py.File(path,'r')[dataset].shape
    return image_volume_shape_map
    
    
def h5_sequential_chunk_writer(prediction_generator,
                               output_volumes, 
                               output_shapes,
                               num_classes,
                               chunk_shape=(32, 64, 64),
                               overlap=(0, 0, 0),
                               min_logit=-0.5,
                               mpi=False):
  '''Sequentially write chunks(with overlap) from volumes'''
  chunk_shape = np.array(chunk_shape)
  chunk_offset = chunk_shape // 2
  overlap = np.array(overlap)
  step_shape = chunk_shape - overlap
  output_volume_map = {}
  
  for vol in output_volumes.split(','):
    volname, path, dataset = vol.split(':')
    if not mpi:
      f = h5py.File(path, 'w')
    else:
      f = h5py.File(path, 'w', driver='mpio', comm=MPI.COMM_WORLD)


    output_shape = output_shapes[volname]
    output_volume_map[volname] = f.create_dataset(name=dataset, shape=output_shape, dtype='float32')
    logits_ds = f.create_dataset(name='logits', shape=list(output_shape)+[num_classes], fillvalue=min_logit, dtype='float32') 
    max_bbox = bounding_box.BoundingBox(start=(0,0,0), size=output_shapes[volname])
    for p in prediction_generator:
      center, logits, pred = p['center'], p['logits'], p['class_prediction']
      pad_w_start = center - chunk_shape // 2 + overlap // 2
      pad_w_end = center + (chunk_shape+1) // 2 - overlap // 2
      coord_offset = overlap // 2
      w_start = pad_w_start - coord_offset
      w_end = pad_w_end - coord_offset
      
      write_bbox = bounding_box.BoundingBox(start=w_start, end=w_end)
      
      write_bbox = bounding_box.intersection(write_bbox, max_bbox)
      read_bbox = bounding_box.BoundingBox(start=coord_offset, size=write_bbox.size)
      
      write_slices = write_bbox.to_slice()
      read_slices = read_bbox.to_slice()
      write_slices = tuple([write_slices[i] for i in [2,1,0]])
      read_slices = tuple([read_slices[i] for i in [2,1,0]])
      output_volume_map[volname][write_slices] = pred[read_slices]
      logits_ds[write_slices] = logits[read_slices]
    f.close()

def h5_sequential_chunk_writer_v2(prediction_generator,
                               output_volumes, 
                               output_shapes,
                               num_classes,
                               chunk_shape=(32, 64, 64),
                               label_shape=(32, 64, 64),
                               overlap=(0, 0, 0),
                               sub_bbox=None,
                               axes='zyx',
                               mpi=False):
  '''Sequentially write chunks(with overlap) from volumes'''
  chunk_shape = np.array(chunk_shape)
  label_shape = np.array(label_shape)
  overlap = np.array(overlap)
  output_volume_map = {}
  
  for vol in output_volumes.split(','):
    volname, path, dataset = vol.split(':')
    if not mpi:
      f = h5py.File(path, 'w')
    else:
      f = h5py.File(path, 'w', driver='mpio', comm=MPI.COMM_WORLD)


    output_shape = output_shapes[volname]
    logging.warn('output_shape %s', output_shape)
    logits_ds = f.create_dataset(name='logits', shape=list(output_shape)+[num_classes], dtype='float32') 
    class_prediction_ds = f.create_dataset(name='class_prediction', shape=output_shape, dtype='float32')
    max_bbox = bounding_box.BoundingBox(start=(0,0,0), size=output_shapes[volname][::-1])
    logging.warn('bbox %s', max_bbox)
    for p in prediction_generator:
      center, logits, class_prediction = p['center'][0], p['logits'], p['class_prediction']

      # deal with initial boarders
      if (center - label_shape // 2 == 0).any():
        r_start = np.array([0,0,0])
        w_start = center - label_shape // 2
        r_size = label_shape
        w_size = label_shape
      else:
        r_start = overlap // 2
        w_start = center - label_shape // 2 + overlap // 2
        r_size = label_shape - overlap // 2
        w_size = label_shape - overlap // 2

      r_slc = np.s_[
        r_start[2]:r_start[2] + r_size[2],
        r_start[1]:r_start[1] + r_size[1],
        r_start[0]:r_start[0] + r_size[0],
      ]
      w_slc = np.s_[
        w_start[2]:w_start[2] + w_size[2],
        w_start[1]:w_start[1] + w_size[1],
        w_start[0]:w_start[0] + w_size[0],
      ]
      logits_ds[w_slc] = logits[r_slc]
      class_prediction_ds[w_slc] = class_prediction[r_slc]


    f.close()

def preprocess_image(image, offset, scale):
  return (tf.cast(image, tf.float32) - offset) / scale

def preprocess_mask_invert(mask):
  '''Membrane as -0.5 and background as 0.5. '''
  return 0.5 - tf.cast(mask, tf.float32)

def crop_v2(tensor, crop_shape, offset=(0, 0, 0), batched=False):
  """Extracts 'crop_shape' around 'offset' from 'tensor'.

  Args:
    tensor: tensor to extract data from (b, [z], y, x, c)
    offset: (x, y, [z]) offset from the center point of 'tensor'; center is
        taken to be 'shape // 2'
    crop_shape: (x, y, [z]) shape to extract
    batched: if True, first dim is b

  Returns:
    cropped tensor
  """
  with tf.name_scope('offset_crop'):
    shape = np.array(tensor.shape.as_list())

    # Nothing to do?
    #if (shape[1:-1] == crop_shape[::-1]).all():
    if shape[1:-1] == crop_shape[::-1]:
      return tensor

    off_y = shape[-3] // 2 - crop_shape[1] // 2 + offset[1]
    off_x = shape[-2] // 2 - crop_shape[0] // 2 + offset[0]

    if not batched:
      tensor = tf.expand_dims(tensor, axis=0)
    if len(offset) == 2:
      cropped = tensor[:,
                       off_y:(off_y + crop_shape[1]),
                       off_x:(off_x + crop_shape[0]),
                       :]
    else:
      off_z = shape[-4] // 2 - crop_shape[2] // 2 + offset[2]
      cropped = tensor[:,
                       off_z:(off_z + crop_shape[2]),
                       off_y:(off_y + crop_shape[1]),
                       off_x:(off_x + crop_shape[0]),
                       :]
    if not batched:
      cropped = tf.squeeze(cropped, axis=0)

    return cropped
def preprocess_mask(mask, label_shape):
  crop_mask = crop_v2(mask, np.zeros((3,), np.int32), np.array(label_shape)[::-1])
  return tf.cast(crop_mask, tf.float32)

def soft_filter(label):
  predicate = tf.greater(tf.reduce_mean(label[...,1:]), 0.2)
  logging.warn('predicate %s', predicate)
  return predicate

def load_from_numpylike(coord_tensor, volume, chunk_shape, volume_axes='zyx'):
  """ Load a chunk from numpylike volume.

  Args:
    coord: Tensor of shape (3,) in xyz order
    volume: A numpy like volume
    chunk_shape: 3-tuple/list of shape in xyz
    volume_axes: specify the axis order in volume
  Returns:
    Tensor loaded with data from coord
  """
  chunk_shape = np.array(chunk_shape)
  def _load_from_numpylike(coord):
    starts = np.array(coord[0]) - chunk_shape // 2
    slc = bounding_box.BoundingBox(start=starts, size=chunk_shape).to_slice()
    if volume_axes == 'zyx':
      data = volume[slc[0], slc[1], slc[2], :]
    elif volume_axes == 'xyz':
      data = volume[slc[2], slc[1], slc[0], :]
      data = data.transpose([2,1,0,3])
      logging.warning('data shape %s', data.shape)
    else:
      raise ValueError('volume_axes mush either be "zyx" or "xyz"')
    return data
  dtype = volume.dtype
  num_classes = volume.shape[-1]
  logging.warn('weird class: %d %s', num_classes, volume.shape)
  with tf.name_scope('load_from_h5') as scope:
    loaded = tf.compat.v1.py_func(
        _load_from_numpylike, [coord_tensor], [dtype],
        name=scope)[0]
    loaded.set_shape(list(chunk_shape[::-1]) + [num_classes])
    logging.warn('after %s', loaded.shape)
    return loaded

def load_from_numpylike_with_pad(coord_tensor, volume, chunk_shape, volume_axes='zyx'):
  """ Load a chunk from numpylike volume.

  Args:
    coord: Tensor of shape (3,) in xyz order
    volume: A numpy like volume
    chunk_shape: 3-tuple/list of shape in xyz
    volume_axes: specify the axis order in volume
  Returns:
    Tensor loaded with data from coord
  """
  chunk_shape = np.array(chunk_shape)
  def _load_from_numpylike(coord):
    starts = np.array(coord[0]) - (chunk_shape-1) // 2
    slc = bounding_box.BoundingBox(start=starts, size=chunk_shape).to_slice()
    logging.warning('loading from %s %s, %s, %s', starts, chunk_shape, slc, volume.shape)
    if volume_axes == 'zyx':
      data = volume[slc[0], slc[1], slc[2], :]
    elif volume_axes == 'xyz':
      data = volume[slc[2], slc[1], slc[0], :]
    else:
      raise ValueError('volume_axes mush either be "zyx" or "xyz"')
    return data
  dtype = volume.dtype
  num_classes = volume.shape[-1]
  logging.warn('weird class: %d %s', num_classes, volume.shape)
  with tf.name_scope('load_from_h5') as scope:
    loaded = tf.py_func(
        _load_from_numpylike, [coord_tensor], [dtype],
        name=scope)[0]
    loaded.set_shape(list(chunk_shape[::-1]) + [num_classes])
    logging.warn('after %s', loaded.shape)
    return loaded
def filter_out_of_bounds(coord, chunk_shape, max_shape):
  logging.warn('filter %s %s %s', coord, chunk_shape, max_shape)
  chunk_shape = np.array(chunk_shape)
  max_shape = np.array(max_shape)
  if True:
    return True
  else:
    logging.error('crap')
    return False
def preprocess_edt_labels(labels):
  labels = tf.cast(labels, tf.float32)
  labels = labels * 10
  return labels

def get_full_size(chunk_size, label_size):
  new_chunk_size = [
    int(np.ceil(chunk_size[0] * np.sqrt(2))),
    int(np.ceil(chunk_size[1] * np.sqrt(2))),
    chunk_size[2]]
  new_label_size = [
    int(np.ceil(label_size[0] * np.sqrt(2))), 
    int(np.ceil(label_size[1] * np.sqrt(2))),
    label_size[2]]
  return new_chunk_size, new_label_size

  

def random_rotate(coord, image, labels, chunk_shape, label_shape):
  angle = np.random.rand() * 2 * np.pi
  image = tfa.image.rotate(image, angle, name='rot_im')
  labels = tfa.image.rotate(labels, angle, name='rot_lb')
  image_offset = np.array(image.shape.as_list())[1:3] // 2 - np.array(chunk_shape)[0:2] // 2
  label_offset = np.array(labels.shape.as_list())[1:3] // 2 - np.array(label_shape)[0:2] // 2
  logging.warning('offsets: %s %s', image_offset, label_offset)
  image= tf.image.crop_to_bounding_box(image, image_offset[0], image_offset[1], chunk_shape[0], chunk_shape[1])
  labels= tf.image.crop_to_bounding_box(labels, label_offset[0], label_offset[1], label_shape[0], label_shape[1])
  logging.warning('post offset shapes: %s %s', image.shape, labels.shape)
  return coord, image, labels

def random_rotate_with_weights(coord, image, labels, weights, chunk_shape, label_shape):
  angle = np.random.rand() * 2 * np.pi
  image = tfa.image.rotate(image, angle, name='rot_im')
  labels = tfa.image.rotate(labels, angle, name='rot_lb')
  weights = tfa.image.rotate(weights, angle, name='rot_im')

  image_offset = np.array(image.shape.as_list())[1:3] // 2 - np.array(chunk_shape)[0:2] // 2
  label_offset = np.array(labels.shape.as_list())[1:3] // 2 - np.array(label_shape)[0:2] // 2

  logging.warning('offsets: %s %s', image_offset, label_offset)
  image= tf.image.crop_to_bounding_box(image, image_offset[0], image_offset[1], chunk_shape[0], chunk_shape[1])
  labels= tf.image.crop_to_bounding_box(labels, label_offset[0], label_offset[1], label_shape[0], label_shape[1])
  weights = tf.image.crop_to_bounding_box(weights, image_offset[0], image_offset[1], chunk_shape[0], chunk_shape[1])
  logging.warning('post offset shapes: %s %s %s', image.shape, labels.shape, weights.shape)
  return coord, image, labels, weights

def train_input_fn(data_volumes, 
                   label_volumes, 
                   tf_coords,
                   num_classes, 
                   chunk_shape, 
                   label_shape, 
                   batch_size, 
                   offset, 
                   scale,
                   rotation=False):
  def h5_coord_chunk_dataset():
    image_volume_map = {}
    for vol in data_volumes.split(','):
      volname, path, dataset = vol.split(':')
      image_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)

    label_volume_map = {}
    for vol in label_volumes.split(','):
      volname, path, dataset = vol.split(':')
      if num_classes > 1:
        label_volume_map[volname] = tf.keras.utils.to_categorical(
          h5py.File(path, 'r')[dataset])
      else:
        label_volume_map[volname] = np.expand_dims(h5py.File(path, 'r')[dataset], axis=-1)
        

    for key in image_volume_map: # Currently only works with one key in the map
      if rotation:
        pre_chunk_shape, pre_label_shape = get_full_size(chunk_shape, label_shape)
      else:
        pre_chunk_shape, pre_label_shape = chunk_shape, label_shape

      logging.warning('pre rotation shapes: %s %s', pre_chunk_shape, pre_label_shape)

      max_shape = image_volume_map[key].shape

      label_scale = 1.0 # convert 0-1 to -0.5, 0.5 for regression model
      label_offset = 0.0 if num_classes > 1 else -0.5 # convert 0-1 to -0.5, 0.5 for regression model


      files = tf.data.Dataset.list_files(tf_coords+'*')
      files = files.shuffle(100)
      ds = files.apply(
        tf.data.experimental.parallel_interleave(
          lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
          cycle_length=2
        )
      )
      ds = ds.shard(hvd.size(), hvd.rank())
      ds = ds.repeat().shuffle(8000)
      ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, volname: (
          coord, 
          load_from_numpylike(coord, image_volume_map[key], pre_chunk_shape), 
          load_from_numpylike(coord, label_volume_map[key], pre_label_shape)), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, image, label: (
          coord, 
          preprocess_image(image, offset, scale),
          tf.cast(label, tf.float32) * label_scale + label_offset), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      if rotation:
        ds = ds.map(lambda coord, image, label: random_rotate(
          coord, image, label, chunk_shape, label_shape), 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)         
      ds = ds.map(lambda coord, image, label: (
        {
        'center': coord[0],
        'image': image
        },
        label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(1)
      logging.warn('ds_shape: %d, %s', batch_size, ds)

      return ds
  return h5_coord_chunk_dataset

def train_input_rebalance_fn(data_volumes, 
                   label_volumes, 
                   tf_coords,
                   num_classes, 
                   chunk_shape, 
                   label_shape, 
                   batch_size, 
                   offset, 
                   scale,
                   rotation=False,
                   rebalance=False
                   ):
  '''
  Rebalance, where to assign stronger weight to foreground
  try to use different loss, like dice loss
  '''
  def get_weight_simple(label):
    '''given label tensor, get weight tensor on the fly'''
    #balance_ratio = tf.reduce_sum(tf.where(label > 0)) * 1.0 / np.product(label.shape)
    #logging.warning('--balance ratio: %s', balance_ratio)
    #tf.print('balance')
    weights = tf.where(label > 0, 100.0, 1.0)
    return weights
  # def get_weight(label_tensor):
  #   def _get_weight_func(label):
  #     valid_ratio = (np.sum(label[:] > 0) / np.product(label_shape))
  #     #balancer = (1 - valid_ratio) / valid_ratio
  #     #logging.warning('--balance ratio: %s %s', valid_ratio, balancer)
  #     weights = np.where(label > 0, 1 - valid_ratio, valid_ratio).astype(np.float32)
  #     return weights

  #   with tf.name_scope('get_weight') as scope:
  #     weights_tensor = tf.compat.v1.py_func(
  #         _get_weight_func, [label_tensor], [tf.float32],
  #         name=scope)[0]
  #     weights_tensor.set_shape(list(label_shape[::-1]) + [num_classes])
  #     logging.warn('after weights %s', weights_tensor.shape)
  #     return weights_tensor


  def h5_coord_chunk_dataset():
    image_volume_map = {}
    for vol in data_volumes.split(','):
      volname, path, dataset = vol.split(':')
      image_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)

    label_volume_map = {}
    for vol in label_volumes.split(','):
      volname, path, dataset = vol.split(':')
      if num_classes > 1:
        label_volume_map[volname] = tf.keras.utils.to_categorical(
          h5py.File(path, 'r')[dataset])
      else:
        label_volume_map[volname] = np.expand_dims(h5py.File(path, 'r')[dataset], axis=-1)
        

    for key in image_volume_map: # Currently only works with one key in the map
      if rotation:
        pre_chunk_shape, pre_label_shape = get_full_size(chunk_shape, label_shape)
      else:
        pre_chunk_shape, pre_label_shape = chunk_shape, label_shape

      logging.warning('pre rotation shapes: %s %s', pre_chunk_shape, pre_label_shape)

      max_shape = image_volume_map[key].shape

      label_scale = 1.0 # convert 0-1 to -0.5, 0.5 for regression model
      # label_offset = 0.0 if num_classes > 1 else -0.5 # convert 0-1 to -0.5, 0.5 for regression model
      label_offset = 0.0


      files = tf.data.Dataset.list_files(tf_coords+'*')
      files = files.shuffle(100)
      ds = files.apply(
        tf.data.experimental.parallel_interleave(
          lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
          cycle_length=2
        )
      )
      ds = ds.shard(hvd.size(), hvd.rank())
      ds = ds.repeat().shuffle(8000)
      ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, volname: (
          coord, 
          load_from_numpylike(coord, image_volume_map[key], pre_chunk_shape), 
          load_from_numpylike(coord, label_volume_map[key], pre_label_shape)), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, image, label: (
          coord, 
          preprocess_image(image, offset, scale),
          tf.cast(label, tf.float32) * label_scale + label_offset), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      if rotation:
        ds = ds.map(lambda coord, image, label: random_rotate(
          coord, image, label, chunk_shape, label_shape), 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)         
      ds = ds.map(lambda coord, image, label: (
        {
          'center': coord[0],
          'image': image,
          'weights': get_weight_simple(label)
        } if rebalance else {
          'center': coord[0],
          'image': image,
        },
        label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(16)
      logging.warn('ds_shape: %d, %s', batch_size, ds)

      return ds
  return h5_coord_chunk_dataset

def train_input_fn_with_weight(
  data_volumes, 
  label_volumes, 
  weight_volumes,
  tf_coords,
  num_classes, 
  chunk_shape, 
  label_shape, 
  batch_size, 
  offset, 
  scale,
  rotation=False):
  def h5_coord_chunk_dataset():
    image_volume_map = {}
    for vol in data_volumes.split(','):
      volname, path, dataset = vol.split(':')
      image_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)

    label_volume_map = {}
    for vol in label_volumes.split(','):
      volname, path, dataset = vol.split(':')
      if num_classes > 1:
        label_volume_map[volname] = tf.keras.utils.to_categorical(
          h5py.File(path, 'r')[dataset])
      else:
        label_volume_map[volname] = np.expand_dims(h5py.File(path, 'r')[dataset], axis=-1)
        
    weight_volume_map = {}
    for vol in weight_volumes.split(','):
      volname, path, dataset = vol.split(':')
      weight_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)
        

    for key in image_volume_map: # Currently only works with one key in the map
      if rotation:
        pre_chunk_shape, pre_label_shape = get_full_size(chunk_shape, label_shape)
      else:
        pre_chunk_shape, pre_label_shape = chunk_shape, label_shape

      logging.warning('pre rotation shapes: %s %s', pre_chunk_shape, pre_label_shape)

      max_shape = image_volume_map[key].shape

      label_scale = 1.0 # convert 0-1 to -0.5, 0.5 for regression model
      label_offset = 0.0 if num_classes > 1 else -0.5 # convert 0-1 to -0.5, 0.5 for regression model


      files = tf.data.Dataset.list_files(tf_coords+'*')
      files = files.shard(hvd.size(), hvd.rank())
      ds = files.apply(
        tf.data.experimental.parallel_interleave(
          lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
          cycle_length=2
        )
      )
      ds = ds.repeat().shuffle(8000)
      ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, volname: (
          coord, 
          load_from_numpylike(coord, image_volume_map[key], pre_chunk_shape), 
          load_from_numpylike(coord, label_volume_map[key], pre_label_shape),
          load_from_numpylike(coord, weight_volume_map[key], pre_chunk_shape),
          ), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, image, label, weights: (
          coord, 
          preprocess_image(image, offset, scale),
          tf.cast(label, tf.float32) * label_scale + label_offset,
          weights
          ), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      if rotation:
        ds = ds.map(lambda coord, image, label, weights: random_rotate_with_weights(
          coord, image, label, weights, chunk_shape, label_shape), 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)         
      ds = ds.map(lambda coord, image, label, weights: (
        {
        'center': coord[0],
        'image': image,
        'weights': weights
        },
        label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(1)
      logging.warn('ds_shape: %d, %s', batch_size, ds)
      return ds

  return h5_coord_chunk_dataset

def load_from_numpylike_mult(
  coord_tensor, 
  volname_tensor, 
  volume_map, 
  chunk_shape, 
  volume_axes='zyx'):
  """ Load a chunk from numpylike volume.

  Args:
    coord: Tensor of shape (3,) in xyz order
    volume: A numpy like volume
    chunk_shape: 3-tuple/list of shape in xyz
    volume_axes: specify the axis order in volume
  Returns:
    Tensor loaded with data from coord
  """
  chunk_shape = np.array(chunk_shape)
  start_offset = (chunk_shape - 1) // 2
  def _load_from_numpylike(coord, volname):
    volume = volume_map[volname.decode('ascii')]
    starts = np.array(coord[0]) - start_offset
    slc = bounding_box.BoundingBox(start=starts, size=chunk_shape).to_slice()
    if volume_axes == 'zyx':
      data = volume[slc[0], slc[1], slc[2], :]
    elif volume_axes == 'xyz':
      data = volume[slc[2], slc[1], slc[0], :]
      data = data.transpose([2,1,0,3])
      logging.warning('data shape %s', data.shape)
    else:
      raise ValueError('volume_axes mush either be "zyx" or "xyz"')
    return data
  volumes = iter(volume_map.values())
  first_vol = next(volumes)
  dtype = first_vol.dtype
  num_classes = first_vol.shape[-1]
  logging.warn('weird class: %d %s', num_classes, first_vol.shape)
  with tf.name_scope('load_from_h5') as scope:
    volname_tensor = tf.squeeze(volname_tensor, axis=0)
    loaded = tf.compat.v1.py_func(
        _load_from_numpylike, [coord_tensor, volname_tensor], [dtype],
        name=scope)[0]
    loaded.set_shape(list(chunk_shape[::-1]) + [num_classes])
    logging.warn('after %s', loaded.shape)
    return loaded

def train_input_mult_fn(
  data_volumes, 
  label_volumes, 
  tf_coords,
  num_classes, 
  chunk_shape, 
  label_shape, 
  batch_size, 
  offset, 
  scale,
  rotation=False,
  rebalance=False,
  label_offset=0.0,
  label_scale=1.0
  ):
  '''
  Multiple sources of input can be used 
  '''
  def h5_coord_chunk_dataset():
    image_volume_map = {}
    for vol in data_volumes.split(','):
      volname, path, dataset = vol.split(':')
      image_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)

    label_volume_map = {}
    for vol in label_volumes.split(','):
      volname, path, dataset = vol.split(':')
      if num_classes > 1:
        label_volume_map[volname] = tf.keras.utils.to_categorical(
          h5py.File(path, 'r')[dataset])
      else:
        label_volume_map[volname] = np.expand_dims(h5py.File(path, 'r')[dataset], axis=-1)
    # logging.warning('input maps %s %s', image_volume_map, label_volume_map)    
    if rotation:
      pre_chunk_shape, pre_label_shape = get_full_size(chunk_shape, label_shape)
    else:
      pre_chunk_shape, pre_label_shape = chunk_shape, label_shape
    # logging.warning('pre rotation shapes: %s %s', pre_chunk_shape, pre_label_shape)

    # chunk_shape and label shape must be the same across volumes
    max_shapes = {}
    for key in image_volume_map:
      max_shapes[key] = image_volume_map[key].shape

    # label_scale = 1.0 # convert 0-1 to -0.5, 0.5 for regression model
    # label_offset = 0.0 if num_classes > 1 else -0.5 # convert 0-1 to -0.5, 0.5 for regression model
    # label_offset = 0.0


    files = tf.data.Dataset.list_files(tf_coords+'*')
    files = files.shuffle(100)
    ds = files.apply(
      tf.data.experimental.parallel_interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=2
      )
    )
    ds = ds.shard(hvd.size(), hvd.rank())
    ds = ds.repeat().shuffle(8000)
    ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda coord, volname: (
        coord, 
        load_from_numpylike_mult(coord, volname, image_volume_map, pre_chunk_shape), 
        load_from_numpylike_mult(coord, volname, label_volume_map, pre_label_shape)), 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda coord, image, label: (
        coord, 
        preprocess_image(image, offset, scale),
        tf.cast(label, tf.float32) * label_scale + label_offset), 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if rotation:
      ds = ds.map(lambda coord, image, label: random_rotate(
        coord, image, label, chunk_shape, label_shape), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)         
    ds = ds.map(lambda coord, image, label: (
      {
        'center': coord[0],
        'image': image,
        'weights': get_weight_simple(label)
      } if rebalance else {
        'center': coord[0],
        'image': image,
      },
      label),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(16)
    logging.warn('ds_shape: %d, %s', batch_size, ds)

    return ds
  return h5_coord_chunk_dataset

def train_input_fn_with_weight(
  data_volumes, 
  label_volumes, 
  weight_volumes,
  tf_coords,
  num_classes, 
  chunk_shape, 
  label_shape, 
  batch_size, 
  offset, 
  scale,
  rotation=False):
  def h5_coord_chunk_dataset():
    image_volume_map = {}
    for vol in data_volumes.split(','):
      volname, path, dataset = vol.split(':')
      image_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)

    label_volume_map = {}
    for vol in label_volumes.split(','):
      volname, path, dataset = vol.split(':')
      if num_classes > 1:
        label_volume_map[volname] = tf.keras.utils.to_categorical(
          h5py.File(path, 'r')[dataset])
      else:
        label_volume_map[volname] = np.expand_dims(h5py.File(path, 'r')[dataset], axis=-1)
        
    weight_volume_map = {}
    for vol in weight_volumes.split(','):
      volname, path, dataset = vol.split(':')
      weight_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)
        

    for key in image_volume_map: # Currently only works with one key in the map
      if rotation:
        pre_chunk_shape, pre_label_shape = get_full_size(chunk_shape, label_shape)
      else:
        pre_chunk_shape, pre_label_shape = chunk_shape, label_shape

      logging.warning('pre rotation shapes: %s %s', pre_chunk_shape, pre_label_shape)

      max_shape = image_volume_map[key].shape

      label_scale = 1.0 # convert 0-1 to -0.5, 0.5 for regression model
      label_offset = 0.0 if num_classes > 1 else -0.5 # convert 0-1 to -0.5, 0.5 for regression model


      files = tf.data.Dataset.list_files(tf_coords+'*')
      files = files.shard(hvd.size(), hvd.rank())
      ds = files.apply(
        tf.data.experimental.parallel_interleave(
          lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
          cycle_length=2
        )
      )
      ds = ds.repeat().shuffle(8000)
      ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, volname: (
          coord, 
          load_from_numpylike(coord, image_volume_map[key], pre_chunk_shape), 
          load_from_numpylike(coord, label_volume_map[key], pre_label_shape),
          load_from_numpylike(coord, weight_volume_map[key], pre_chunk_shape),
          ), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(lambda coord, image, label, weights: (
          coord, 
          preprocess_image(image, offset, scale),
          tf.cast(label, tf.float32) * label_scale + label_offset,
          weights
          ), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      if rotation:
        ds = ds.map(lambda coord, image, label, weights: random_rotate_with_weights(
          coord, image, label, weights, chunk_shape, label_shape), 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)         
      ds = ds.map(lambda coord, image, label, weights: (
        {
        'center': coord[0],
        'image': image,
        'weights': weights
        },
        label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(1)
      logging.warn('ds_shape: %d, %s', batch_size, ds)
      return ds

  return h5_coord_chunk_dataset

# Manipulate CloudVolumes
def ffn_to_cv(ffn_bb):
  '''Convert ffn style bbox to cloudvolume style.'''
  offset = np.array(ffn_bb.start)
  size = np.array(ffn_bb.size)
  return Bbox(a=offset, b=offset+size)

def get_chunk_bboxes(
  union_bbox, 
  chunk_size, 
  overlap,
  include_small_sub_boxes=True,
  back_shift_small_sub_boxes=False):
  ffn_style_bbox = bounding_box.BoundingBox(
    np.array(union_bbox.minpt), np.array(union_bbox.size2()))

  calc = bounding_box.OrderlyOverlappingCalculator(
    outer_box=ffn_style_bbox, 
    sub_box_size=chunk_size, 
    overlap=overlap, 
    include_small_sub_boxes=include_small_sub_boxes,
    back_shift_small_sub_boxes=back_shift_small_sub_boxes)

  bbs = [ffn_to_cv(ffn_bb) for ffn_bb in calc.generate_sub_boxes()]

  return bbs

def prepare_precomputed(precomputed_path, offset, size, 
  resolution, chunk_size, factor=(1,2,1), layer_type='segmentation', dtype='uint32'):
  cv_args = dict(
    bounded=False, fill_missing=True, autocrop=False,
    cache=False, compress_cache=None, cdn_cache=False,
    progress=False, provenance=None, compress=True, 
    non_aligned_writes=True, parallel=False)
  if layer_type == 'image':
    encoding = 'raw'
    compress = False
  elif layer_type == 'segmentation':
    encoding = 'compressed_segmentation'
    compress = True
  else:
    raise ValueError
  info = CloudVolume.create_new_info(
    num_channels=0,
    layer_type=layer_type,
    data_type=dtype,
    encoding=encoding,
    # compress=compress,
    resolution=list(resolution),
    voxel_offset=np.array(offset),
    volume_size=np.array(size),
    chunk_size=chunk_size,
    max_mip=-1,
    factor=factor,
    )
  cv = CloudVolume('file://'+precomputed_path, mip=-1, info=info, **cv_args)
  cv.commit_info()
  return cv