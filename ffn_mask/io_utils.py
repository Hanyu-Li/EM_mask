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
  # print(sample_start, sample_size)
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
  # examples = tf.parse_single_example(proto, features=dict(
      center=tf.io.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
      # center=tf.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
      label_volume_name=tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
  ))
  coord = examples['center']
  volname = examples['label_volume_name']
  return coord, volname

def h5_sequential_chunk_generator(data_volumes, 
                                  chunk_shape=(32,64, 64), # xyz
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
      # print('step:', step_shape, step_counts)
      # pad_data = np.pad(val, zip(begin_pad, end_pad), 'reflect')
      # print(pad_data.shape)
      grid_zyx = [np.arange(j)*i+d//2 for i,j,d in zip(step_shape, step_counts, chunk_shape)]
      # print('grid:', grid_zyx)
      grid = np.array(np.meshgrid(*grid_zyx)).T.reshape(-1, 3)
      
      for i in range(grid.shape[0]):
        center = grid[i]
        image = _load_from_numpylike_with_pad(center, val, pad_start, pad_end, 
          chunk_shape, sample_start, sample_size )
        #logging.warn('loaded_shape %s', image.shape)

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
      # print('step:', step_shape, step_counts)
      # pad_data = np.pad(val, zip(begin_pad, end_pad), 'reflect')
      # print(pad_data.shape)
      grid_zyx = [np.arange(j)*i+d//2 for i,j,d in zip(step_shape, step_counts, chunk_shape)]
      # print('grid:', grid_zyx)
      grid = np.array(np.meshgrid(*grid_zyx)).T.reshape(-1, 3)
      
      for i in range(grid.shape[0]):
        center = grid[i]
        image = _load_from_numpylike_with_pad(center, val, pad_start, pad_end, 
          chunk_shape, sample_start, sample_size )
        # logging.warn('loaded_shape %s', image.shape)

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
    #z,y,x = h5py.File(path,'r')[dataset].shape
    #image_volume_shape_map[volname] = [z,y,x,num_classes]
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
    # logging.warn('output_shape %s', output_shape)
    output_volume_map[volname] = f.create_dataset(name=dataset, shape=output_shape, dtype='float32')
    logits_ds = f.create_dataset(name='logits', shape=list(output_shape)+[num_classes], fillvalue=min_logit, dtype='float32') 
    # output_volume = np.zeros(shape=output_shapes[volname], dtype=np.float32)
    max_bbox = bounding_box.BoundingBox(start=(0,0,0), size=output_shapes[volname])
    for p in prediction_generator:
      center, logits, pred = p['center'], p['logits'], p['class_prediction']
      # logging.warn('pred shape %s', pred.shape)
      pad_w_start = center - chunk_shape // 2 + overlap // 2
      pad_w_end = center + (chunk_shape+1) // 2 - overlap // 2
      coord_offset = overlap // 2
      w_start = pad_w_start - coord_offset
      w_end = pad_w_end - coord_offset
      # logging.warn('diagnose_1 %s %s %s', center, pad_w_start, coord_offset)
      # logging.warn('diagnose_2 %s %s', w_start, w_end)
      
      write_bbox = bounding_box.BoundingBox(start=w_start, end=w_end)
      # logging.warn('write_bbox %s, %s', center, write_bbox.size)
      
      write_bbox = bounding_box.intersection(write_bbox, max_bbox)
      read_bbox = bounding_box.BoundingBox(start=coord_offset, size=write_bbox.size)
      
      write_slices = write_bbox.to_slice()
      read_slices = read_bbox.to_slice()
      write_slices = tuple([write_slices[i] for i in [2,1,0]])
      read_slices = tuple([read_slices[i] for i in [2,1,0]])
      # logging.warn('pred shape %s', pred[read_slices].shape)
      output_volume_map[volname][write_slices] = pred[read_slices]
      logits_ds[write_slices] = logits[read_slices]
#       output_volume[write_slices] = pred[read_slices+(0,)]
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
  # chunk_offset = chunk_shape // 2
  # overlap = np.array(overlap)
  # if axes == 'zyx':
  #   chunk_shape = chunk_shape[::-1]
  #   chunk_offset = chunk_offset[::-1]
    # overlap = chunk_shape[::-1]
  # step_shape = chunk_shape - overlap
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
    # output_volume = np.zeros(shape=output_shapes[volname], dtype=np.float32)
    max_bbox = bounding_box.BoundingBox(start=(0,0,0), size=output_shapes[volname][::-1])
    logging.warn('bbox %s', max_bbox)
    for p in prediction_generator:
      center, logits, class_prediction = p['center'][0], p['logits'], p['class_prediction']
      # logging.warn('pred shape %s %s', center, pred.shape)
      # pad_w_start = center - chunk_shape // 2 + overlap // 2
      # pad_w_end = center + (chunk_shape+1) // 2 - overlap // 2
      # coord_offset = overlap // 2
      # w_start = pad_w_start - coord_offset
      # w_end = pad_w_end - coord_offset
      # logging.warn('diagnose_1 %s %s %s', center, pad_w_start, coord_offset)
      # logging.warn('diagnose_2 %s %s', w_start, w_end)
      
      # write_bbox = bounding_box.BoundingBox(start=w_start, end=w_end)
      # logging.warn('write_bbox %s, %s', center, write_bbox.size)
      

      # write_bbox = bounding_box.intersection(write_bbox, max_bbox)
      # read_bbox = bounding_box.BoundingBox(start=coord_offset, size=write_bbox.size)
      
      # write_slices = write_bbox.to_slice()
      # read_slices = read_bbox.to_slice()
      # write_slices = tuple([write_slices[i] for i in [2,1,0]])
      # read_slices = tuple([read_slices[i] for i in [2,1,0]])

      # r_start = chunk_shape // 2 - label_shape // 2 + overlap // 2

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
      # logging.warning('io: %s, %s, %s, %s, %s', center, r_start, r_size, w_start, w_size)

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
      # logging.warning('slc: %s, %s', r_slc, w_slc)
      # print(logits.shape)
      # logits_ds[w_slc] = logits[r_slc]
      # class_prediction_ds[w_slc] = class_prediction[r_slc]
      logits_ds[w_slc] = logits[r_slc]
      class_prediction_ds[w_slc] = class_prediction[r_slc]


      # logging.warn('pred shape %s', pred[read_slices].shape)
      # output_volume_map[volname][write_slices] = pred[read_slices]
#       output_volume[write_slices] = pred[read_slices+(0,)]
    f.close()

def preprocess_image(image, offset, scale):
  return (tf.cast(image, tf.float32) - offset) / scale

def preprocess_mask_invert(mask):
  '''Membrane as -0.5 and background as 0.5. '''
  return 0.5 - tf.cast(mask, tf.float32)
def preprocess_mask(mask, label_shape):
  logging.warn('>> preprocess_mask %s %s', mask.shape, label_shape)
  #crop_mask = crop_v2(mask, mask.shape, label_shape)
  crop_mask = crop_v2(mask, np.zeros((3,), np.int32), np.array(label_shape)[::-1])
  return tf.cast(crop_mask, tf.float32)
  # return tf.cast(mask, tf.float32) - 0.5

def soft_filter(label):
  #predicate = tf.logical_or(
  #  tf.greater(tf.reduce_mean(label[...,1:]), 0.8),
  #  tf.greater(tf.random_uniform((1,)), 0.95))
  #logging.warn('predicate %s', predicate)
  #return predicate[0]
  predicate = tf.greater(tf.reduce_mean(label[...,1:]), 0.2)
  logging.warn('predicate %s', predicate)
  return predicate

def train_input_fn_old(data_volumes, label_volumes, num_classes, chunk_shape, label_shape, batch_size, offset, scale):
  '''An tf.data.Dataset from h5 file containing'''
  h5_chunk_gen = h5_random_chunk_generator(data_volumes, label_volumes, num_classes, chunk_shape)
  ds = tf.data.Dataset.from_generator(
    h5_chunk_gen, (tf.int64, tf.uint8, tf.bool), 
    (tf.TensorShape((3,)), tf.TensorShape(chunk_shape+(1,)), tf.TensorShape(chunk_shape+(num_classes,))))
  ds = ds.map(lambda x,y,z: (x, preprocess_image(y, offset, scale),
                             preprocess_mask(z, label_shape)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x,y,z: soft_filter(z))
  ds = ds.repeat().batch(batch_size)
  ds = ds.shard(hvd.size(), hvd.rank())
        
  value = ds.make_one_shot_iterator().get_next()
  # print('>>>', len(value))
  features = {
    'center': value[0],
    'image': value[1]
  }
  labels = value[2]
  return features, labels

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
  # chunk_shape_xyz = np.array(chunk_shape[::-1])
  chunk_shape = np.array(chunk_shape)
  def _load_from_numpylike(coord):
    # starts = np.array(coord[0]) - (chunk_shape-1) // 2
    starts = np.array(coord[0]) - chunk_shape // 2
    slc = bounding_box.BoundingBox(start=starts, size=chunk_shape).to_slice()
    # logging.warning('slices %s', slc)
    # slc is in z,y,x order
    # logging.warn('loading from %s %s, %s, %s', starts, chunk_shape, slc, volume.shape)
    if volume_axes == 'zyx':
      data = volume[slc[0], slc[1], slc[2], :]
    elif volume_axes == 'xyz':
      data = volume[slc[2], slc[1], slc[0], :]
      data = data.transpose([2,1,0,3])
      logging.warning('data shape %s', data.shape)
    else:
      raise ValueError('volume_axes mush either be "zyx" or "xyz"')
    # logging.warn('loaded data %s ', data.shape)
    # assert (data.shape[1:4] == chunk_shape[::-1]).all()
    # logging.error('shape_check %s, %s', data.shape[0:3], chunk_shape[::-1])
    return data
  dtype = volume.dtype
  num_classes = volume.shape[-1]
  logging.warn('weird class: %d %s', num_classes, volume.shape)
  with tf.name_scope('load_from_h5') as scope:
    loaded = tf.compat.v1.py_func(
        _load_from_numpylike, [coord_tensor], [dtype],
        name=scope)[0]
    # logging.warn('before %s', loaded.shape)
    # loaded.set_shape([1] + list(chunk_shape[::-1]) + [num_classes])
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
  # chunk_shape_xyz = np.array(chunk_shape[::-1])
  chunk_shape = np.array(chunk_shape)
  def _load_from_numpylike(coord):
    starts = np.array(coord[0]) - (chunk_shape-1) // 2
    slc = bounding_box.BoundingBox(start=starts, size=chunk_shape).to_slice()
    # slc is in z,y,x order
    logging.warning('loading from %s %s, %s, %s', starts, chunk_shape, slc, volume.shape)
    if volume_axes == 'zyx':
      data = volume[slc[0], slc[1], slc[2], :]
    elif volume_axes == 'xyz':
      data = volume[slc[2], slc[1], slc[0], :]
    else:
      raise ValueError('volume_axes mush either be "zyx" or "xyz"')
    # logging.warn('loaded data %s ', data.shape)
    # assert (data.shape[1:4] == chunk_shape[::-1]).all()
    # logging.error('shape_check %s, %s', data.shape[0:3], chunk_shape[::-1])
    return data
  dtype = volume.dtype
  num_classes = volume.shape[-1]
  logging.warn('weird class: %d %s', num_classes, volume.shape)
  with tf.name_scope('load_from_h5') as scope:
    loaded = tf.py_func(
        _load_from_numpylike, [coord_tensor], [dtype],
        name=scope)[0]
    # logging.warn('before %s', loaded.shape)
    # loaded.set_shape([1] + list(chunk_shape[::-1]) + [num_classes])
    loaded.set_shape(list(chunk_shape[::-1]) + [num_classes])
    logging.warn('after %s', loaded.shape)
    return loaded
def filter_out_of_bounds(coord, chunk_shape, max_shape):
  logging.warn('filter %s %s %s', coord, chunk_shape, max_shape)
  # if image.shape.as_list()[1:4] == list(chunk_shape)[::-1]:
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
  # with tf.device('/cpu:0'):
  # image = tf.contrib.image.rotate(image, angle, name='rot_im')
  image = tfa.image.rotate(image, angle, name='rot_im')
  labels = tfa.image.rotate(labels, angle, name='rot_lb')
  image_offset = np.array(image.shape.as_list())[1:3] // 2 - np.array(chunk_shape)[0:2] // 2
  label_offset = np.array(labels.shape.as_list())[1:3] // 2 - np.array(label_shape)[0:2] // 2
  # off_x = shape[-2] // 2 - crop_shape[0] // 2 + offset[0]
  logging.warning('offsets: %s %s', image_offset, label_offset)
  image= tf.image.crop_to_bounding_box(image, image_offset[0], image_offset[1], chunk_shape[0], chunk_shape[1])
  labels= tf.image.crop_to_bounding_box(labels, label_offset[0], label_offset[1], label_shape[0], label_shape[1])
  logging.warning('post offset shapes: %s %s', image.shape, labels.shape)
  return coord, image, labels

def random_rotate_with_weights(coord, image, labels, weights, chunk_shape, label_shape):
  angle = np.random.rand() * 2 * np.pi
  # with tf.device('/cpu:0'):
  # image = tf.contrib.image.rotate(image, angle, name='rot_im')
  image = tfa.image.rotate(image, angle, name='rot_im')
  labels = tfa.image.rotate(labels, angle, name='rot_lb')
  weights = tfa.image.rotate(weights, angle, name='rot_im')

  image_offset = np.array(image.shape.as_list())[1:3] // 2 - np.array(chunk_shape)[0:2] // 2
  label_offset = np.array(labels.shape.as_list())[1:3] // 2 - np.array(label_shape)[0:2] // 2
  # weights_offset = np.array(weights.shape.as_list())[1:3] // 2 - np.array(chunk_shape)[0:2] // 2

  # off_x = shape[-2] // 2 - crop_shape[0] // 2 + offset[0]
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
      # label_scale = 1.0 if num_classes > 1 else 2.0 # convert 0-1 to -0.5, 0.5 for regression model
      # label_offset = 0.0 # convert 0-1 to -0.5, 0.5 for regression model

      label_scale = 1.0 # convert 0-1 to -0.5, 0.5 for regression model
      label_offset = 0.0 if num_classes > 1 else -0.5 # convert 0-1 to -0.5, 0.5 for regression model

      # fnames = tf.matching_files(tf_coords+'*')
      # logging.info('fnames %s', fnames)
      # ds = tf.data.TFRecordDataset(fnames, compression_type='GZIP')

      files = tf.data.Dataset.list_files(tf_coords+'*')
      # ds = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'))
      # files = files.shard(hvd.size(), hvd.rank())
      files = files.shuffle(100)
      # files = files.apply(
      #   tf.data.experimental.filter_for_shard(hvd.size(), hvd.rank()))
      ds = files.apply(
        tf.data.experimental.parallel_interleave(
          lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
          cycle_length=2
        )
      )
      # ds = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), cycle_length=2)
      ds = ds.shard(hvd.size(), hvd.rank())
      ds = ds.repeat().shuffle(8000)
      ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # ds = ds.filter(lambda coord, volname: filter_out_of_bounds(coord, chunk_shape, max_shape))
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
      # label_scale = 1.0 if num_classes > 1 else 2.0 # convert 0-1 to -0.5, 0.5 for regression model
      # label_offset = 0.0 # convert 0-1 to -0.5, 0.5 for regression model

      label_scale = 1.0 # convert 0-1 to -0.5, 0.5 for regression model
      label_offset = 0.0 if num_classes > 1 else -0.5 # convert 0-1 to -0.5, 0.5 for regression model

      # fnames = tf.matching_files(tf_coords+'*')
      # logging.info('fnames %s', fnames)
      # ds = tf.data.TFRecordDataset(fnames, compression_type='GZIP')

      files = tf.data.Dataset.list_files(tf_coords+'*')
      # ds = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'))
      files = files.shard(hvd.size(), hvd.rank())
      # files = files.apply(
      #   tf.data.experimental.filter_for_shard(hvd.size(), hvd.rank()))
      ds = files.apply(
        tf.data.experimental.parallel_interleave(
          lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
          cycle_length=2
        )
      )
      # ds = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), cycle_length=2)
      ds = ds.repeat().shuffle(8000)
      ds = ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # ds = ds.filter(lambda coord, volname: filter_out_of_bounds(coord, chunk_shape, max_shape))
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

def predict_input_fn(data_volumes, chunk_shape, 
  overlap, batch_size, offset, scale, bounding_box, var_threshold):
  '''An tf.data.Dataset from h5 file containing'''
  h5_chunk_gen = h5_sequential_chunk_generator(data_volumes, 
    chunk_shape, overlap, bounding_box, var_threshold)
  ds = tf.data.Dataset.from_generator(
    generator=h5_chunk_gen, 
    output_types=(tf.int64, tf.uint8), 
    output_shapes=(tf.TensorShape((3,)), 
                   tf.TensorShape(chunk_shape+(1,))))
  ds = ds.map(lambda x,y: (x, preprocess_image(y, offset, scale)))
  ds = ds.batch(batch_size)
  ds = ds.shard(hvd.size(), hvd.rank())
        
  value = ds.make_one_shot_iterator().get_next()
  features = {
    'center': value[0],
    'image': value[1]
  }
  return features

def predict_input_fn_v2(data_volumes,
                        chunk_shape,
                        label_shape,
                        overlap,
                        batch_size,
                        offset,
                        scale,
                        sub_bbox, 
                        var_threshold):

  if sub_bbox is not None:
    bbox = bounding_box_pb2.BoundingBox()
    text_format.Parse(sub_bbox, bbox)
  else:
    # in x, y, z
    start = (0, 0, 0)
    size = next(iter(get_h5_shapes(data_volumes).values()))
    logging.warn('shapes: %s %s', start, size)
    bbox = bounding_box.BoundingBox(start=start, size=size[::-1])

  logging.warning('bbox: %s', bbox)
  # in/out shape discrepency due to size shrinking after conv pool layers
  in_out_diff = (np.array(chunk_shape) - np.array(label_shape)) // 2
  overlap_padded = overlap + in_out_diff

  image_volume_map = {}
  for vol in data_volumes.split(','):
    volname, path, dataset = vol.split(':')
    image_volume_map[volname] = np.expand_dims(h5py.File(path,'r')[dataset], axis=-1)
  # assume only one volname key in inference


  def h5_sequential_bbox_gen():
    calc = bounding_box.OrderlyOverlappingCalculator(
      outer_box=bbox, 
      sub_box_size=chunk_shape, 
      overlap=overlap_padded, 
      include_small_sub_boxes=True,
      back_shift_small_sub_boxes=True)
    for bb in calc.generate_sub_boxes():
      yield [bb.start + bb.size // 2] # central coord which is io'ed

  ds = tf.data.Dataset.from_generator(
    generator=h5_sequential_bbox_gen, 
    output_types=(tf.int64), 
    output_shapes=(tf.TensorShape((1,3)))
  )
  ds = ds.apply(
    tf.data.experimental.filter_for_shard(hvd.size(), hvd.rank()))
  ds = ds.map(lambda coord: (
      coord, 
      load_from_numpylike(coord, image_volume_map[volname], chunk_shape)),
     num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda coord, image: (coord, preprocess_image(image, offset, scale)),
     num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # )
    # num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda coord, image:
    {
      'center': coord,
      'image': image
    },
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)
  return ds

def get_h5_shape(input_path, axes='zyx'):
  volname, path, dataset = input_path.split(':')
  with h5py.File(path, 'r'):
    shape = f[dataset].shape
    if axes == 'zyx':
      shape = shape[::-1]
    elif axes == 'xyz':
      pass
    else:
      raise ValueError('must be zyx or xyz')
    return shape
    


def predict_input_fn_h5(input_volume,
                        input_offset, 
                        input_size,
                        chunk_shape,
                        label_shape,
                        overlap,
                        batch_size,
                        offset,
                        scale,
                        var_threshold):

  volname, path, dataset = input_volume.split(':')
  if input_offset is None or input_size is None:
    with h5py.File(path, 'r') as f:
      input_size = f[dataset].shape[::-1]
    input_offset = (0, 0, 0)
  slc = np.s_[
    input_offset[2]:input_offset[2]+input_size[2],
    input_offset[1]:input_offset[1]+input_size[1],
    input_offset[0]:input_offset[0]+input_size[0],
    ]
  logging.warn('slc: %s', slc)

  with h5py.File(path, 'r') as f:
    data = np.expand_dims(f[dataset][slc], axis=-1)
  logging.warning('input_shape %s', data.shape)
  # this bbox coord is relative to offset 
  bbox = bounding_box.BoundingBox(start=[0,0,0], size=input_size)

  in_out_diff = (np.array(chunk_shape) - np.array(label_shape)) // 2
  overlap_padded = overlap + in_out_diff
  def h5_sequential_bbox_gen():
    calc = bounding_box.OrderlyOverlappingCalculator(
      outer_box=bbox, 
      sub_box_size=chunk_shape, 
      overlap=overlap_padded, 
      include_small_sub_boxes=True,
      back_shift_small_sub_boxes=True)
    for bb in calc.generate_sub_boxes():
      yield [bb.start + bb.size // 2] # central coord which is io'ed

  ds = tf.data.Dataset.from_generator(
    generator=h5_sequential_bbox_gen, 
    output_types=(tf.int64), 
    output_shapes=(tf.TensorShape((1,3)))
  )
  ds = ds.apply(
    tf.data.experimental.filter_for_shard(hvd.size(), hvd.rank()))
  ds = ds.map(lambda coord: (
      coord, 
      load_from_numpylike(coord, data, chunk_shape)),
     num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda coord, image: (coord, preprocess_image(image, offset, scale)),
     num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # )
    # num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda coord, image:
    {
      'center': coord,
      'image': image
    },
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)
  return ds


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

def crop_v3(tensor, crop_shape, offset=(0, 0, 0), batched=False):
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
      tensor = tensor[:,
                       off_y:(off_y + crop_shape[1]),
                       off_x:(off_x + crop_shape[0]),
                       :]
    else:
      off_z = shape[-4] // 2 - crop_shape[2] // 2 + offset[2]
      tensor = tensor[:,
                       off_z:(off_z + crop_shape[2]),
                       off_y:(off_y + crop_shape[1]),
                       off_x:(off_x + crop_shape[0]),
                       :]
    if not batched:
      tensor = tf.squeeze(tensor, axis=0)

    return tensor

# 2D image utils 
def generate_train_valid_dataset(
  train_set,
  valid_set,
):
  pass