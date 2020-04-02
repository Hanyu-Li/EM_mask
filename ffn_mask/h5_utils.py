"""H5 input fn for prediction. """
import numpy as np
import h5py
import logging
import tensorflow as tf
from ffn.utils import bounding_box
from .precomputed_utils import get_bboxes
from ffn_mask.io_utils import preprocess_image
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
    # starts = np.array(coord[0]) - (chunk_shape-1) // 2
    starts = np.array(coord[0]) - chunk_shape // 2
    # bbox = Bbox(a=starts, b=starts+chunk_shape)
    # width = np.array(chunk_shape) // 2
    # logging.warning('py_func: %s %s', coord, width)
    # slc = np.s_[coord[0] - width[0]:coord[0] + width[0],
    #   coord[1] - width[1]:coord[1] + width[1],
    #   coord[2] - width[2]:coord[2] + width[2]
    # ]
    slc = np.s_[starts[0]:starts[0] + chunk_shape[0],
               starts[1]:starts[1] + chunk_shape[1],
               starts[2]:starts[2] + chunk_shape[2]]
    if volume_axes == 'xyz':
      data = np.expand_dims(volume[slc][...], axis=-1)
    if volume_axes == 'zyx':
      slc = slc[::-1]
      # logging.warning('load slc: %s', slc)
      data = np.expand_dims(volume[slc][...], axis=-1)
      # logging.warning('load chunk: %s', data.shape)
    else:
      raise ValueError('volume_axes mush either be "zyx" or "xyz"')
    return data

  dtype = volume.dtype
  if len(volume.shape) == 3:
    num_classes = 1
  else:
    num_classes = volume.shape[-1]
  # logging.warn('weird class: %d %s', num_classes, volume.shape)
  with tf.name_scope('load_from_h5') as scope:
    loaded = tf.compat.v1.py_func(
        _load_from_numpylike, [coord_tensor], [dtype],
        name=scope)[0]
    # logging.warn('before %s', loaded.shape)
    loaded.set_shape(list(chunk_shape[::-1]) + [num_classes])
    # loaded.set_shape(list(chunk_shape[::]) + [num_classes])
    # logging.warn('after %s', loaded.shape)
    return loaded
def predict_input_fn_h5_v2(
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

  # with h5py.File(path, 'r') as f:
  #   # data = np.expand_dims(f[dataset][slc], axis=-1)
  #   data = np.expand_dims(f[dataset][slc], axis=-1)

  f_in = h5py.File(path, 'r')
  data = f_in[dataset]

  logging.warning('data_shape %s', data.shape)
  # this bbox coord is relative to offset 
  # union_bbox = bounding_box.BoundingBox(start=[0,0,0], size=input_size)

  if mpi_rank == 0:
    union_bbox = bounding_box.BoundingBox(start=input_offset, size=input_size)
    sub_bboxes = get_bboxes(union_bbox, chunk_size=chunk_shape, 
      overlap=overlap, back_shift_small=True, backend='ffn')
    ranked_sub_bboxes = np.array_split(sub_bboxes, mpi_size)
  else:
    ranked_sub_bboxes = None

  ranked_sub_bboxes = mpi_comm.scatter(ranked_sub_bboxes, 0)
  
  logging.warning('num_sub_bbox %d %s', len(ranked_sub_bboxes), ranked_sub_bboxes[0])
  # logging.warning('bbox %s %s', ranked_sub_bboxes[0].minpt, ranked_sub_bboxes[0].maxpt)
  logging.warning('bbox %s %s', ranked_sub_bboxes[0].start, ranked_sub_bboxes[0].end)
  def sub_bbox_iterator():
    for sb in ranked_sub_bboxes:
      logging.warning('load bbox %s', (sb.start + sb.end) // 2)
      yield [(sb.start + sb.end) // 2]
  # in_out_diff = (np.array(chunk_shape) - np.array(label_shape)) // 2
  # overlap_padded = overlap + in_out_diff

  ds = tf.data.Dataset.from_generator(
    generator=sub_bbox_iterator, 
    output_types=(tf.int64), 
    output_shapes=(tf.TensorShape((1,3)))
  )
  # ds = ds.apply(
  #   tf.data.experimental.filter_for_shard(hvd.size(), hvd.rank()))
  ds = ds.map(lambda coord: (
      coord, 
      load_from_h5(coord, data, chunk_shape)),
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

def get_h5_shape(data_volume):
  volname, path, dataset = data_volume.split(':')
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
  sub_bbox=None,
  axes='zyx',
  mpi=True):
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
  
  # for vol in output_volumes.split(','):
  volname, path, dataset = output_volume.split(':')
  if not mpi:
    f = h5py.File(path, 'w')
  else:
    f = h5py.File(path, 'w', driver='mpio', comm=MPI.COMM_WORLD)

  # output_shape = output_shapes[volname]
  # output_shape = output_shapes[volname]
  output_size = output_size[::-1]
  logging.warn('output_shape %s', output_size)
  logits_ds = f.create_dataset(name='logits', shape=list(output_size)+[num_classes], 
    fillvalue=0,
    dtype='float32') 
  class_prediction_ds = f.create_dataset(name='class_prediction', shape=output_size, 
    fillvalue=0, dtype='float32')
  # output_volume = np.zeros(shape=output_shapes[volname], dtype=np.float32)
  # max_bbox = bounding_box.BoundingBox(start=(0,0,0), size=output_size[::-1])
  # logging.warn('bbox %s', max_bbox)
  write_size = chunk_shape  - overlap
  zyx_overlap = overlap[::-1]
  logging.warning('write_size %s', write_size)
  # for p in tqdm(prediction_generator, total=3080):
  for p in prediction_generator:
    # center, logits, class_prediction = p['center'], p['logits'], p['class_prediction']
    # logging.warn('pred shape %s %s', center, class_prediction.shape)
    # logging.warn('pred result: %s', np.mean(logits[:]))
    # logging.warn('pred %s %s %s', center, logits.shape, np.mean(logits[:]))
    for center, logits, class_prediction in zip(p['center'], p['logits'], p['class_prediction']):
    # for center, logits, class_prediction in zip(p['center'], p['logits'], p['class_prediction']):
      logging.warn('pred %s %s %s', center, logits.shape, np.mean(logits[:]))
      zyx_center = center[0][::-1]
      zyx_write_size = write_size[::-1]

      zyx_start = zyx_center - zyx_write_size // 2
      # zyx_write_rad = zyx_write_size // 2
      # zyx_write_rad[zyx_write_rad == 0] = 1

      w_slc = np.s_[
        zyx_start[0] : zyx_start[0] + zyx_write_size[0], 
        zyx_start[1] : zyx_start[1] + zyx_write_size[1], 
        zyx_start[2] : zyx_start[2] + zyx_write_size[2]]
      r_slc = np.s_[
        zyx_overlap[0] // 2: zyx_overlap[0] // 2 + zyx_write_size[0],
        zyx_overlap[1] // 2: zyx_overlap[1] // 2 + zyx_write_size[1],
        zyx_overlap[2] // 2: zyx_overlap[2] // 2 + zyx_write_size[2],
      ]
      logging.warning('write slc %s read slc %s', w_slc, r_slc)
      logging.warning('pred_center %s, write_center %s', center, zyx_start + zyx_write_size // 2)
      logging.warning('overlap %s', zyx_overlap)
      logging.warning('writing shape: %s', logits[r_slc].shape)
      logits_ds[w_slc] = np.array(logits[r_slc])
      class_prediction_ds[w_slc] = class_prediction[r_slc]
  f.close()

      # zyx_center = center

#   for p in prediction_generator:
#     center, logits, class_prediction = p['center'][0], p['logits'], p['class_prediction']
#     logging.warn('pred shape %s %s', center, class_prediction.shape)

#     # deal with initial boarders
#     if (center - label_shape // 2 == 0).any():
#       r_start = np.array([0,0,0])
#       w_start = center - label_shape // 2
#       r_size = label_shape
#       w_size = label_shape
#     else:
#       r_start = overlap // 2
#       w_start = center - label_shape // 2 + overlap // 2
#       r_size = label_shape - overlap // 2
#       w_size = label_shape - overlap // 2
#     # logging.warning('io: %s, %s, %s, %s, %s', center, r_start, r_size, w_start, w_size)

#     r_slc = np.s_[
#       r_start[2]:r_start[2] + r_size[2],
#       r_start[1]:r_start[1] + r_size[1],
#       r_start[0]:r_start[0] + r_size[0],
#     ]
#     w_slc = np.s_[
#       w_start[2]:w_start[2] + w_size[2],
#       w_start[1]:w_start[1] + w_size[1],
#       w_start[0]:w_start[0] + w_size[0],
#     ]
#     logging.warning('slc: %s, %s', r_slc, w_slc)
#     # print(logits.shape)
#     # logits_ds[w_slc] = logits[r_slc]
#     # class_prediction_ds[w_slc] = class_prediction[r_slc]
#     logits_ds[w_slc] = logits[r_slc]
#     class_prediction_ds[w_slc] = class_prediction[r_slc]


#     # logging.warn('pred shape %s', pred[read_slices].shape)
#     # output_volume_map[volname][write_slices] = pred[read_slices]
# #       output_volume[write_slices] = pred[read_slices+(0,)]