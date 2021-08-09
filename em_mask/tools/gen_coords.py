'''
Due to sparseness of synapse, training examples are generated semi automatically
'''
import h5py
import numpy as np
import tensorflow as tf

import sys
import os
from tqdm.auto import tqdm

from scipy import ndimage
import logging

import argparse
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def gen_coords(in_path, out_path, lom_radius, margin, n_samples, true_ratio):
  def get_all_coords():
    name, path, dataset = in_path.split(':')
    with h5py.File(path, 'r') as f:
      print(f.keys())
      label = f[dataset][...]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask = label > 0


    # dilate xy, and z separately
    xy_delta = lom_radius[1]
    z_delta = lom_radius[0]
    xy_struct = ndimage.generate_binary_structure(3, 1)
    z_struct = ndimage.generate_binary_structure(3, 1)
    xy_struct[0, :, :] = False
    xy_struct[2, :, :] = False
    z_struct[:, 0, :] = False
    z_struct[:, 2, :] = False
    z_struct[:, :, 0] = False
    z_struct[:, :, 2] = False
    mask_xy = ndimage.binary_dilation(mask, structure=xy_struct, iterations=xy_delta)
    mask_both = ndimage.binary_dilation(mask_xy, structure=z_struct, iterations=z_delta)

    slc = np.s_[
      margin[0]:mask.shape[0]-margin[0],
      margin[1]:mask.shape[1]-margin[1],
      margin[2]:mask.shape[2]-margin[2]
    ]
    true_coords = np.stack(np.where(mask_both[slc] > 0), axis=0).T + margin
    false_coords = np.stack(np.where(mask_both[slc] == 0), axis=0).T + margin
    
    print(true_coords.shape)
    print(false_coords.shape)

    n_true = np.floor(n_samples * true_ratio).astype(np.int32)
    n_false = n_samples - n_true
    print(n_true)
    print(n_false)

    true_samples = np.random.choice(true_coords.shape[0], size=n_true, replace=False)
    false_samples = np.random.choice(false_coords.shape[0], size=n_false, replace=False)
    tc = true_coords[true_samples]
    fc = false_coords[false_samples]
    all_coords = np.concatenate([tc, fc], axis=0)
    np.random.shuffle(all_coords)
    print(all_coords.shape)
    all_inds = np.arange(all_coords.shape[0])
    print(all_inds.shape)
    return all_coords
  
  if mpi_rank == 0:
    all_coords = get_all_coords()
    all_inds = np.arange(all_coords.shape[0])
    subset_inds = np.array_split(all_inds, mpi_size)
  else:
    all_coords = None
    subset_inds = None
  
  all_coords = mpi_comm.bcast(all_coords, 0)
  subset_inds = mpi_comm.scatter(subset_inds, 0)

  name, path, dataset = in_path.split(':')
  logging.info('Shape %d %s.', mpi_rank, subset_inds.shape)

  sharded_fname = '%s-%s-of-%s' % (out_path, str(mpi_rank).zfill(5), str(mpi_size).zfill(5))
  logging.info(sharded_fname)

  record_options = tf.io.TFRecordOptions(
      tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
  with tf.io.TFRecordWriter(sharded_fname, options=record_options) as writer:
    # for i, coord_idx in tqdm(subset_inds):
    for i in tqdm(subset_inds):
      # z, y, x = np.unravel_index(coord_idx, vol_shapes[i])
      z, y, x = all_coords[i]

      coord = tf.train.Example(features=tf.train.Features(feature=dict(
          center=_int64_feature([x, y, z]),
          label_volume_name=_bytes_feature(name.encode('utf-8'))
      )))
      writer.write(coord.SerializeToString())

  
def gen_multi_coords(in_path, out_path, lom_radius, margin, n_samples, true_ratio):
  def get_all_coords():
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_coords = []
    all_names = []
    for i, partvol in enumerate(in_path.split(',')):
      name, path, dataset = partvol.split(':')
      with h5py.File(path, 'r') as f:
        # print(f.keys())
        label = f[dataset][...]
      mask = label > 0


      # dilate xy, and z separately
      xy_delta = lom_radius[1]
      z_delta = lom_radius[0]
      xy_struct = ndimage.generate_binary_structure(3, 1)
      z_struct = ndimage.generate_binary_structure(3, 1)
      xy_struct[0, :, :] = False
      xy_struct[2, :, :] = False
      z_struct[:, 0, :] = False
      z_struct[:, 2, :] = False
      z_struct[:, :, 0] = False
      z_struct[:, :, 2] = False
      mask_xy = ndimage.binary_dilation(mask, structure=xy_struct, iterations=xy_delta)
      mask_both = ndimage.binary_dilation(mask_xy, structure=z_struct, iterations=z_delta)

      slc = np.s_[
        margin[0]:mask.shape[0]-margin[0],
        margin[1]:mask.shape[1]-margin[1],
        margin[2]:mask.shape[2]-margin[2]
      ]
      true_coords = np.stack(np.where(mask_both[slc] > 0), axis=0).T + margin
      false_coords = np.stack(np.where(mask_both[slc] == 0), axis=0).T + margin
      
      print(true_coords.shape)
      print(false_coords.shape)

      n_true = np.floor(n_samples * true_ratio).astype(np.int32)
      n_false = n_samples - n_true
      print(n_true)
      print(n_false)

      true_samples = np.random.choice(true_coords.shape[0], size=n_true, replace=False)
      false_samples = np.random.choice(false_coords.shape[0], size=n_false, replace=False)
      tc = true_coords[true_samples]
      fc = false_coords[false_samples]

      coords = np.concatenate([tc, fc], axis=0)
      np.random.shuffle(coords)
      # all_coords = np.concatenate([all_coords, coords], axis=0)
      all_coords.extend(list(coords))
      # print(all_coords.shape)
      all_names.extend([name] * len(coords))
      # all_inds = np.arange(all_coords.shape[0])
      # print(all_inds.shape)
    # print(all_coords, all_names)
    return all_coords, all_names
  
  if mpi_rank == 0:
    all_coords, all_names = get_all_coords()
    # all_inds = np.arange(all_coords.shape[0])
    all_inds = np.arange(len(all_coords))
    np.random.shuffle(all_inds)
    subset_inds = np.array_split(all_inds, mpi_size)
  else:
    all_coords = None
    all_names = None
    subset_inds = None
  
  all_coords = mpi_comm.bcast(all_coords, 0)
  all_names = mpi_comm.bcast(all_names, 0)
  subset_inds = mpi_comm.scatter(subset_inds, 0)

  # name, path, dataset = in_path.split(':')
  logging.info('Shape %d %s.', mpi_rank, subset_inds.shape)

  sharded_fname = '%s-%s-of-%s' % (out_path, str(mpi_rank).zfill(5), str(mpi_size).zfill(5))
  logging.info(sharded_fname)

  record_options = tf.io.TFRecordOptions(
      tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
  with tf.io.TFRecordWriter(sharded_fname, options=record_options) as writer:
    # for i, coord_idx in tqdm(subset_inds):
    for i in tqdm(subset_inds):
      # z, y, x = np.unravel_index(coord_idx, vol_shapes[i])
      z, y, x = all_coords[i]
      name = all_names[i]
      # logging.warning('record %s, at %d %d %d', name, z, y, x)

      coord = tf.train.Example(features=tf.train.Features(feature=dict(
          center=_int64_feature([x, y, z]),
          label_volume_name=_bytes_feature(name.encode('utf-8'))
      )))
      writer.write(coord.SerializeToString())

  
  
  




def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--in_path', type=str, default=None)
  parser.add_argument('--out_path', type=str, default=None)
  # parser.add_argument('--dataset_name', type=str, default='label')
  parser.add_argument('--lom_radius', type=str, default='6,40,40')
  parser.add_argument('--margin', type=str, default='16,128,128')
  parser.add_argument('--n_samples', type=int, default=None)
  parser.add_argument('--true_ratio', type=float, default=0.8)
  
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  lom_radius = [int(i) for i in args.lom_radius.split(',')]
  margin = [int(i) for i in args.margin.split(',')]
  gen_multi_coords(args.in_path, args.out_path,
    lom_radius, margin, args.n_samples, args.true_ratio)

if __name__ == '__main__':
  main()