''' Merge Input Function and related tools.'''
import numpy as np
import cloudvolume
from cloudvolume.lib import Bbox
# import neuroglancer
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
import scipy
import skimage
from scipy import ndimage
from skimage.feature import peak_local_max, corner_peaks
from skimage.morphology import local_maxima
from skimage.segmentation import random_walker, watershed
from ffn.utils.bounding_box import OrderlyOverlappingCalculator, BoundingBox
from scipy.spatial.distance import cdist, euclidean
# from scipy.stats import pearsonr

from tqdm import tqdm
import logging
import tensorflow as tf
import horovod.tensorflow as hvd

from .precomputed_utils import load_from_precomputed

import edt
from pprint import pprint

def loc_to_bbox(loc, size):
  loc = np.array(loc)
  size= np.array(size)
  start = loc - size // 2
  end = start + size
  bb = Bbox(a=start, b=end)
  return bb

def find_3d_median(points, resolution=(6,6,40)):
  '''Find pseudo median as point closest to 3d mean'''
  resolution = np.array(resolution)
  phys_points = points * resolution
  
  mean_pos = np.mean(phys_points, axis=0, keepdims=True)
  dists = cdist(phys_points, mean_pos)
  median_pos = points[np.argmin(dists)]
  return median_pos

def parse_neighbor(value, coord, 
  min_voxel_count=2000, offset=(0, 0, 0), 
  resolution=(6,6,40), 
  max_sample=1,
  visited=set()):
  '''Get non-zero value and median points'''
  offset = np.array(offset)
  uni, count = np.unique(value, return_counts=True)
  results = []
  delete_id = [i for i in uni if i in visited]
  assert 0 not in uni
  delete_idx = np.where(np.isin(uni, delete_id))
  uni = np.delete(uni, delete_idx)
  count = np.delete(count, delete_idx)
  candidates = uni[count > min_voxel_count]
  if len(candidates):
    select = np.random.choice(np.arange(len(candidates)), max_sample, replace=False)
    for c in candidates[select]:
      voxels = coord[value == c, :]
      size = voxels.shape[0]
      median_pos = find_3d_median(voxels)
      results.append({
        'neighbor_id': c,
        'size': size, 
        'median_pos': median_pos + offset})
  return results
  

def training_data_prep(
  seg_chunk, 
  # rad=(16, 16, 8), 
  size_thresh=500, 
  resolution=(12,12,40), 
  offset=(0, 0, 0),
  rescale=(1, 1, 1)):
  '''
  For each seg_chunk, chose center object if  > size_thresh within a central sub window.
  '''
  offset = np.array(offset)
  rescale = np.array(rescale) # for easier localization in neuroglancer
  x, y, z = seg_chunk.shape
  src = seg_chunk[x//2, y//2, z//2]
  src_mask = seg_chunk == src
  src_size = np.sum(src_mask[:])
  if src_size < size_thresh:
    return {}
  
  neighbor_dict = dict()
  src_mask_dilate = ndimage.binary_dilation(src_mask, structure=np.ones((3,3,3)), iterations=1)
  border_mask = skimage.segmentation.find_boundaries(src_mask_dilate, connectivity=2, mode='outer')
  border_mask = np.logical_and(border_mask, seg_chunk != 0)
  border_value = seg_chunk[border_mask]
  border_coord = np.stack(np.where(border_mask), axis=1)

  res = parse_neighbor(border_value, border_coord, 
    min_voxel_count=500, resolution=resolution, max_sample=1)
  for r in res:
    neighbor_dict[(src, r['neighbor_id'])] = {
      'size': r['size'],
      'relative_pos': r['median_pos'],
      'global_pos': (r['median_pos'] + offset) * rescale
    }
  # return neighbor_dict
  neighbor_dict = build_training_data(seg_chunk, neighbor_dict)
  # neighbor_dict = neighbor_dict.update(mask_dict)
  # in_out_pairs = build_training_data(seg_chunk, neighbor_dict)
  return neighbor_dict

def build_training_data(seg_chunk, candidate_dict):
  # factor = [2, 2, 1]
#   bin_struct = np.zeros((8,8,8))

  # in_out_pairs = []
  # in_out_dict = {}
  for k, v in candidate_dict.items():
    # print(k, v)
    uni = np.unique(seg_chunk)
    in_mask = np.isin(seg_chunk, k)
    # print(in_mask.shape) 
    bridge = np.zeros_like(seg_chunk, dtype=np.uint8)
    bridge[v['relative_pos'][0], 
           v['relative_pos'][1], 
           v['relative_pos'][2]] = 1
    bin_struct = ndimage.generate_binary_structure(3, 1)
    bridge = ndimage.binary_dilation(bridge, structure=bin_struct, iterations=4)
    fuse_mask = np.logical_or(in_mask, bridge).astype(np.uint8)
    out_mask = (seg_chunk == k[0]).astype(np.uint8)

    bridge_mask = ndimage.binary_dilation(bridge, structure=np.ones((3,3,3)), iterations=2)
    bridge_mask = np.logical_and(bridge_mask, fuse_mask).astype(np.uint8)
    bridge_mask = (ndimage.filters.gaussian_filter(bridge_mask * 10, sigma=1) > 0).astype(np.uint8)
    # in_out_pairs.append((fuse_mask, out_mask))
    candidate_dict[k] = {
      'fuse_mask': fuse_mask, 
      'correct_mask': out_mask,
      'bridge_mask': bridge_mask
    }

  return candidate_dict



def build_dataset_from_cv(
  coord, 
  seg_cv, 
  chunk_size, 
  resolution,
  factor,
  ):
  def _build_dataset_from_cv(coord):
    coord = coord[0]
    bb = loc_to_bbox(coord, chunk_size)
    seg_chunk = np.array(seg_cv[bb][..., 0])
    offset = coord - chunk_size // 2
    center_id = seg_chunk[chunk_size[0] // 2, chunk_size[1]//2, chunk_size[2]//2]
    # center_id = np.expand_dims(center_id, axis=0)
    # if center_id == 0:
    #   return np.empty(1)
    pair_dict = training_data_prep(
      seg_chunk,
      resolution=resolution,
      offset=offset,
      rescale=factor
    )
    # print(pair_dict)
    # in_out_dict = build_training_data(seg_chunk, pair_dict)

    # logging.warning('>> out len %d', len(in_out_dict))
    if not len(pair_dict):
      dummy = np.zeros((0, chunk_size[0], chunk_size[1], chunk_size[2]), dtype=np.uint8)
      return center_id, center_id, dummy, dummy, 0

    # batch_input, batch_output = zip(*in_out)
    # batch_input = np.stack(batch_input, axis=0)
    # batch_output = np.stack(batch_output, axis=0)
    # return center_id, batch_input, batch_output, len(in_out)

    results = []
    for k, v in pair_dict.items():
      results.append(
        (k[0], k[1], v['fuse_mask'], v['bridge_mask'])
      )

    batch_src, batch_tgt, batch_input, batch_output = zip(*results)
    batch_src = np.stack(batch_src, axis=0)
    batch_tgt = np.stack(batch_tgt, axis=0)
    batch_input = np.stack(batch_input, axis=0)
    batch_output = np.stack(batch_output, axis=0)
    return batch_src, batch_tgt, batch_input, batch_output, len(pair_dict)



  with tf.name_scope('convert_seg_to_mask') as scope:
    center_id, neighbor_id, in_mask, out_mask, n_sample = tf.compat.v1.py_func(
      _build_dataset_from_cv, [coord], 
      [tf.uint32, tf.uint32, tf.uint8, tf.uint8, tf.int64],
      name=scope)
    # in_mask.set_shape(list(chunk_shape[::-1]) + [num_classes])
    in_mask.set_shape([1] + list(chunk_size))
    out_mask.set_shape([1] + list(chunk_size))
    # logging.warning('>> %s', mask)
    return center_id, neighbor_id, in_mask, out_mask, n_sample

# def flatten_examples()

def merge_input_fn(
  seg_cv,
  chunk_size,
  factor,
  n_samples=100,
  batch_size=1
  ):
  # def _merge_input_fn():
  chunk_size = np.array(chunk_size)
  x, y, z = chunk_size
  mip = seg_cv.mip
  resolution = seg_cv.resolution
  global_bounds = seg_cv.meta.bounds(mip) 
  minpt = global_bounds.minpt + chunk_size // 2
  maxpt = global_bounds.maxpt - chunk_size // 2
  r = tf.random.stateless_uniform(
    [n_samples, 1, 3], (4, 2)
  )
  ds = tf.data.Dataset.from_tensor_slices(r)
  ds = ds.shard(hvd.size(), hvd.rank())
  ds = ds.map(
    lambda rand_num: tf.cast(rand_num * (maxpt - minpt) + minpt, tf.int32)
  )
  ds = ds.map(
    lambda coord: (
      coord,
      load_from_precomputed(coord, seg_cv, [1,1,1], volume_axes='zyx')),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(
    lambda coord, center: tf.greater(center[0, 0, 0, 0], 0)
  )
  ds = ds.map(lambda coord, _: (
      coord,
      build_dataset_from_cv(coord, seg_cv, chunk_size, resolution=resolution, factor=factor)),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda coord, masks: tf.greater(masks[4], 0))
  ds = ds.map(lambda coord, masks: (
    {
      'center': coord,
      'id_a': masks[0],
      'id_b': masks[1],
      'image': tf.cast(tf.expand_dims(tf.transpose(masks[2][0], (2, 1, 0)), axis=-1), tf.float32)
    },
    tf.cast(tf.expand_dims(tf.transpose(masks[3][0], (2, 1, 0)), axis=-1), tf.float32)
  ))
  ds = ds.batch(batch_size).prefetch(16)

  return ds
