"""H5 input fn for prediction. """
import numpy as np
import h5py
import logging
import tensorflow as tf
from ffn.utils import bounding_box
from .precomputed_utils import get_bboxes
from em_mask.io_utils import preprocess_image
from mpi4py import MPI
from pprint import pprint
from tqdm import tqdm
import os
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def load_from_h5(coord_tensor, volume, chunk_shape, volume_axes='zyx'):
  """ Load a chunk from precomputed volume with border check.

  Args:
    coord: Tensor of shape (3,) in xyz order
    volume: An opened h5 dataset
    chunk_shape: 3-tuple/list of shape in xyz
    volume_axes: specify the axis order in volume
  Returns:
    Tensor loaded with data from coord
  """
  chunk_shape = np.array(chunk_shape)
  def _load_from_numpylike(coord):
    starts = np.array(coord[0]) - chunk_shape // 2
    slc = np.s_[starts[0]:starts[0] + chunk_shape[0],
               starts[1]:starts[1] + chunk_shape[1],
               starts[2]:starts[2] + chunk_shape[2]]
    if volume_axes == 'xyz':
      data = np.expand_dims(volume[slc][...], axis=-1)
    if volume_axes == 'zyx':
      slc = slc[::-1]
      data = np.expand_dims(volume[slc][...], axis=-1)
    else:
      raise ValueError('volume_axes mush either be "zyx" or "xyz"')
    return data

  dtype = volume.dtype
  if len(volume.shape) == 3:
    num_classes = 1
  else:
    num_classes = volume.shape[-1]
  with tf.name_scope('load_from_h5') as scope:
    loaded = tf.compat.v1.py_func(
        _load_from_numpylike, [coord_tensor], [dtype],
        name=scope)[0]
    loaded.set_shape(list(chunk_shape[::-1]) + [num_classes])
    return loaded

def get_num_of_bbox(input_offset, input_size, chunk_shape, overlap):
  union_bbox = bounding_box.BoundingBox(start=input_offset, size=input_size)
  sub_bboxes = get_bboxes(union_bbox, chunk_size=chunk_shape, 
    overlap=overlap, back_shift_small=True, backend='ffn')
  return len(sub_bboxes)

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

def predict_input_fn_h5(
  input_volume,
  input_offset, 
  input_size,
  chunk_shape,
  label_shape,
  overlap,
  batch_size,
  offset,
  scale,
  var_threshold):
  """MPI inference of h5.
  
  For incoming h5 volume, break down into sub bboxes, and use subsets according
  to mpi rank
  """
  # volname, path, dataset = input_volume.split(':')
  path, dataset = input_volume.split(':')
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

  f_in = h5py.File(path, 'r')
  data = f_in[dataset]

  logging.warning('data_shape %s', data.shape)
  # this bbox coord is relative to offset 

  if mpi_rank == 0:
    union_bbox = bounding_box.BoundingBox(start=input_offset, size=input_size)
    sub_bboxes = get_bboxes(union_bbox, chunk_size=chunk_shape, 
      overlap=overlap, back_shift_small=True, backend='ffn')
    ranked_sub_bboxes = np.array_split(sub_bboxes, mpi_size)
  else:
    ranked_sub_bboxes = None

  ranked_sub_bboxes = mpi_comm.scatter(ranked_sub_bboxes, 0)
  
  logging.warning('num_sub_bbox %d %s', len(ranked_sub_bboxes), ranked_sub_bboxes[0])
  logging.warning('bbox %s %s', ranked_sub_bboxes[0].start, ranked_sub_bboxes[0].end)
  def sub_bbox_iterator():
    for sb in ranked_sub_bboxes:
      yield [(sb.start + sb.end) // 2]

  ds = tf.data.Dataset.from_generator(
    generator=sub_bbox_iterator, 
    output_types=(tf.int64), 
    output_shapes=(tf.TensorShape((1,3)))
  )
  ds = ds.map(lambda coord: (
      coord, 
      load_from_h5(coord, data, chunk_shape)),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda coord, image: (coord, preprocess_image(image, offset, scale)),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda coord, image:
    {
      'center': coord,
      'image': image
    },
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)
  return ds

def get_h5_shape(data_volume):
  # volname, path, dataset = data_volume.split(':')
  path, dataset = data_volume.split(':')
  return h5py.File(path,'r')[dataset].shape

def h5_mpi_writer(
  prediction_generator,
  output_volume,
  output_size,
  num_classes,
  output_offset=(0, 0, 0),
  chunk_shape=(32, 64, 64),
  label_shape=(32, 64, 64),
  overlap=(0, 0, 0),
  num_iter=None,
  axes='zyx',
  mpi=True):
  '''Sequentially write chunks(with overlap) from volumes'''
  chunk_shape = np.array(chunk_shape)
  label_shape = np.array(label_shape)
  overlap = np.array(overlap)
  output_volume_map = {}
  
  # volname, path, dataset = output_volume.split(':')
  # output_path
  if not mpi:
    f = h5py.File(output_volume, 'w')
  else:
    f = h5py.File(output_volume, 'w', driver='mpio', comm=MPI.COMM_WORLD)

  output_size = output_size[::-1]
  logging.warn('output_shape %s', output_size)
  logits_ds = f.create_dataset(name='logits', shape=list(output_size)+[num_classes], 
    fillvalue=-0.5,
    dtype='float32') 
  class_prediction_ds = f.create_dataset(name='class_prediction', shape=output_size, 
    fillvalue=0, dtype='float32')
  write_size = chunk_shape  - overlap
  zyx_overlap = overlap[::-1]
  logging.warning('write_size %s', write_size)
  for p in tqdm(prediction_generator, total=num_iter):
    for center, logits, class_prediction in zip(p['center'], p['logits'], p['class_prediction']):
      zyx_center = center[0][::-1] - output_offset[::-1]
      zyx_write_size = write_size[::-1]

      zyx_start = zyx_center - zyx_write_size // 2

      w_slc = np.s_[
        zyx_start[0] : zyx_start[0] + zyx_write_size[0], 
        zyx_start[1] : zyx_start[1] + zyx_write_size[1], 
        zyx_start[2] : zyx_start[2] + zyx_write_size[2]]
      r_slc = np.s_[
        zyx_overlap[0] // 2: zyx_overlap[0] // 2 + zyx_write_size[0],
        zyx_overlap[1] // 2: zyx_overlap[1] // 2 + zyx_write_size[1],
        zyx_overlap[2] // 2: zyx_overlap[2] // 2 + zyx_write_size[2],
      ]
      assert logits_ds[w_slc].shape == logits[r_slc].shape
      logits_ds[w_slc] = np.array(logits[r_slc])
      class_prediction_ds[w_slc] = class_prediction[r_slc]
  f.close()