'''Given VC and SJ predictions, perform synapse pairing.'''
from copy import copy
import numpy as np
import neuroglancer
import networkx as nx
import pickle
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
import json
from scipy import ndimage
import h5py
from collections import defaultdict
import matplotlib.pyplot as plt
from ffn.utils import bounding_box
from pprint import pprint
import pandas as pd
from tqdm.auto import tqdm
import logging
import argparse
import os
tqdm.pandas()

from skimage.segmentation import find_boundaries, watershed
import fastremap

from mpi4py import MPI
tqdm.monitor_interval = 0
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

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
  '''Use ffn bbox calculator to generate overlapping cloudvolume bbox.'''
  ffn_style_bbox = bounding_box.BoundingBox(
    np.array(union_bbox.minpt), np.array(union_bbox.size3()))

  calc = bounding_box.OrderlyOverlappingCalculator(
    outer_box=ffn_style_bbox, 
    sub_box_size=chunk_size, 
    overlap=overlap, 
    include_small_sub_boxes=include_small_sub_boxes,
    back_shift_small_sub_boxes=back_shift_small_sub_boxes)

  bbs = [ffn_to_cv(ffn_bb) for ffn_bb in calc.generate_sub_boxes()]
  return bbs

def get_pos(seg_id, vc_id, overlap, offset):
  pos = np.stack(np.where(np.logical_and(overlap[..., 0] == seg_id, overlap[..., 1] == vc_id)), axis=1)
  med_pos = np.median(pos, axis=0)
  return med_pos + offset

# def keep_max(df):
#   idx_0 = df.groupby(['seg_id'], sort=False)['vc_size'].transform(max) == df['vc_size']
#   idx_1 = df.groupby(['vc_id'], sort=False)['vc_size'].transform(max) == df['vc_size']
#   return df[idx_0 & idx_1]

def keep_max_generic(df, group_name, value_name):
  '''Remove dup group_name rows and keep the max according to value_name'''
  idx = df.groupby([group_name], sort=False)[value_name].transform(max) == df[value_name]
  return df[idx]

# def keep_max(df):
#   idx_0 = df.groupby(['seg_id'], sort=False)['vc_size'].transform(max) == df['vc_size']
#   idx_1 = df.groupby(['vc_id'], sort=False)['vc_size'].transform(max) == df['vc_size']
#   return df[idx_0 & idx_1]

def get_all_poses(seg_id, vc_id, overlap, offset):
  sel_mask = np.logical_and(overlap[..., 0] == seg_id, overlap[..., 1] == vc_id)
  # mean_val = np.mean(overlap[sel_mask, 2])
  # print(mean_val)
  pos = np.stack(np.where(sel_mask), axis=1)
  med_pos = np.median(pos, axis=0) + offset
  min_pos = np.min(pos, axis=0) + offset
  max_pos = np.max(pos, axis=0) + offset
  return med_pos, min_pos, max_pos

def find_vc(seg_chunk, mask_chunk, vc_chunk, offset, vc_thresh=30, size_thresh=500):
  '''For seg_chunk, find all pre synaptic vc sites and find matching post synaptic partner'''
  # mask chunk 1 soma 2 vessel
  vc_chunk[np.isin(mask_chunk, [1, 2])] = 0
  vc_chunk = ndimage.gaussian_filter(vc_chunk, sigma=(2, 2, 1))
  # vc_chunk = ndimage.median_filter(vc_chunk, sigma=(2, 2, 1))
  # vc_chunk = ndimage.gaussian_filter(vc_chunk, sigma=(1, 1, 0.5))

  vc_chunk[seg_chunk == 0] = 0

  vc_mask = vc_chunk > vc_thresh
  vc_seeds, _ = ndimage.label(vc_chunk > vc_thresh * 2)
  vc_labels = watershed(-vc_chunk, markers=vc_seeds, mask=vc_chunk > vc_thresh, 
    connectivity=np.ones((3,3,3)))
  
  # vc_labels, _ = ndimage.label(vc_mask)
  overlap = np.stack([seg_chunk, vc_labels], axis=-1)
  valid_overlaps = overlap[np.logical_and(overlap[..., 0] != 0, overlap[..., 1] != 0), :]
  if not valid_overlaps.shape[0]:
    return None, vc_labels
  uni_pairs, uni_counts = np.unique(valid_overlaps, axis=0, return_counts=True)
  

  pair_count_entry = [
    {'seg_id': k[0],
    'vc_id': k[1],
    'vc_size': v} for k, v in zip(uni_pairs, uni_counts) if k[0] != 0 and k[1] != 0 and v > size_thresh]
  
  if not len(pair_count_entry):
    # return None, vc_labels, sj_labels
    return None, vc_labels

  
  seg_vc_df = pd.DataFrame(pair_count_entry)
  # seg_vc_df = keep_max(seg_vc_df)
  seg_vc_df = keep_max_generic(seg_vc_df, 'vc_id', 'vc_size')
  # seg_vc_df['vc_pos'] = seg_vc_df.progress_apply(lambda row: get_pos(row.seg_id, row.vc_id, overlap, offset), axis=1)
  # seg_vc_df['vc_pos'], seg_vc_df['vc_min_pos'], seg_vc_df['vc_max_pos'] = zip(*seg_vc_df.progress_apply(
  seg_vc_df['vc_pos'], seg_vc_df['vc_min_pos'], seg_vc_df['vc_max_pos'] = zip(*seg_vc_df.apply(
    lambda row: get_all_poses(row.seg_id, row.vc_id, overlap, offset), axis=1))
  # seg_vc_df['vc_pos'] = seg_vc_df.apply(lambda row: get_pos(row.seg_id, row.vc_id, overlap, offset).tolist(), axis=1)
  
  # return seg_vc_df, vc_labels, sj_labels
  return seg_vc_df, vc_labels

  
def get_neighbors(mask, labels, border_thickness=(5, 5, 2), min_size=100, max_neighbor_count=3):
  '''For a binary mask, find it's neibhor ids in labels.'''
  bin_struct = ndimage.generate_binary_structure(3, 1)
  xy_struct = bin_struct.copy()
  xy_struct[:, :, 0] = 0
  xy_struct[:, :, 2] = 0
  z_struct = np.zeros_like(bin_struct)
  z_struct[1, 1, :] = 1
  
#   mask_border = ndimage.binary_dilation(mask, structure=struct, iterations=border_thickness)
  # mask_border = ndimage.binary_dilation(mask, structure=xy_struct, iterations=border_thickness)
  # mask_border = ndimage.binary_dilation(mask_border, structure=z_struct, iterations=border_thickness // 2)
  mask_border = ndimage.binary_dilation(mask, structure=xy_struct, iterations=border_thickness[0])
  mask_border = ndimage.binary_dilation(mask_border, structure=z_struct, iterations=border_thickness[2])
  mask_border[mask > 0] = 0
  labels_on_border = np.where(mask_border > 0, labels, 0)
  uni, counts = np.unique(labels_on_border, return_counts=True)
  valid = np.logical_and(uni != 0, counts > min_size)
  uni, counts = uni[valid], counts[valid]
  
  if len(uni) > max_neighbor_count:
    order = np.argsort(counts)[::-1]
    uni, counts = uni[order][:max_neighbor_count], counts[order][:max_neighbor_count]
  
  results = []
  for u, c in zip(uni, counts):
    pos = np.median(np.stack(np.where(labels_on_border == u), axis=1), axis=0)
    results.append({
      'id': u,
      'size': c,
      'pos': pos
    })
  return results
def find_sj(seg_vc_df, seg_chunk, vc_labels, mask_chunk, sj_chunk, offset, 
            sj_thresh=30, pad=(3, 3, 2), border_thickness=(5, 5, 2), min_sj_size=25, max_neighbor_count=3):
  '''Find sj partner for each vc'''
  line_annos = []
  offset = np.array(offset)
  # rad = np.array(rad)
  pad = np.array(pad) # pad bbox around vc object
  valid_pair_entries = []
  seg_ids_with_vc = np.array((seg_vc_df['seg_id']))
  
  sj_chunk[np.isin(mask_chunk, [1, 2])] = 0
  sj_chunk = ndimage.gaussian_filter(sj_chunk, sigma=(1, 1, 0))
  # sj_mask = sj_chunk > sj_thresh * 2
  # sj_labels, _ = ndimage.label(sj_mask)
  # sj_labels = 
  sj_seeds, _ = ndimage.label(sj_chunk > sj_thresh * 2)
  sj_labels = watershed(-sj_chunk, markers=sj_seeds, mask=sj_chunk > sj_thresh, 
    connectivity=np.ones((3,3,3)))

#   input_bb = Bbox(offset, offset + np.array(seg_chunk.shape))
  input_bb = Bbox((0, 0, 0), seg_chunk.shape)
  # pad = np.array([3, 3, 2])
  
  for ind, row in tqdm(seg_vc_df.iterrows(), total=len(seg_vc_df), disable=True):
    pos = (row.vc_pos - offset).astype(np.int32)

    vc_bbox = Bbox(row.vc_min_pos - offset - pad, row.vc_max_pos - offset + pad)
    # print(vc_bbox)
    # inter_bb = Bbox.intersection(input_bb, Bbox(pos - rad, pos + rad))
    inter_bb = Bbox.intersection(input_bb, vc_bbox)
    if np.product(inter_bb.size3()) == 0: 
      continue
    local_slc = inter_bb.to_slices()
    local_offset = offset + inter_bb.minpt
#     local_sj = sj_chunk[local_slc]
      
#     local_seg = seg_chunk[local_slc]
#     local_mask = local_seg == row.seg_id
    local_vc_labels = vc_labels[local_slc]
    local_vc_mask = local_vc_labels == row.vc_id
    local_sj_chunk = sj_chunk[local_slc]
    local_sj_labels = sj_labels[local_slc]
    
#     sj_entries = get_sj_candidates(local_vc_mask, local_sj_labels, border_thickness, min_sj_size)
    sj_entries = get_neighbors(local_vc_mask, local_sj_labels, 
      border_thickness, min_sj_size, max_neighbor_count)
    # if row.vc_id == 594:
    #   pprint(sj_entries)
    if not len(sj_entries):
      continue
    for s in sj_entries:
      # if s['id'] == 317:
#       logging.warning('Identified synapse %d, %d, %s, %s', row.seg_id, s['sj_id'], s['sj_pos'], s['sj_size'])
      mean_sj_value = np.mean(local_sj_chunk[local_sj_labels == s['id']])
      # print('mean', mean_sj_value)
      norm_size = s['size'] * min(mean_sj_value / 128.0, 1.0)
      if norm_size < min_sj_size:
        continue
      valid_pair_entries.append({
        'pre_seg_id': row.seg_id,
        'vc_id': row.vc_id,
#         'post_seg_id': ordered_ids[0],
        'vc_pos': row.vc_pos,
        'vc_size': row.vc_size,
        'sj_id': s['id'],
        'sj_pos': s['pos'] + local_offset,
        'sj_size': s['size'],
        'sj_norm_size': norm_size,
        'sj_value': mean_sj_value
      })
  synapse_df = pd.DataFrame(valid_pair_entries)
  if not len(synapse_df):
    return None, sj_labels
  synapse_df = keep_max_generic(synapse_df, 'sj_id', 'sj_size')
  # line_annos = [neuroglancer.LineAnnotation(
  #   point_a=row.vc_pos,
  #   point_b=row.sj_pos,
  #   id=ind) for ind, row in synapse_df.iterrows()]
  return synapse_df, sj_labels


def get_angle(a, b, c):
  ba = a - b
  bc = c - b
  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)
  return np.degrees(angle)


def find_post_syn(synapse_df, seg_chunk, sj_labels, offset, 
  rad=(50, 50, 10), max_angle=90.0, border_thickness=(3, 3, 2)):
  input_bb = Bbox((0, 0, 0), seg_chunk.shape)
  synapse_entries = []
  pre_seg_ids = synapse_df.index
  for ind, row in tqdm(synapse_df.iterrows(), total=len(synapse_df), disable=True):
    pos = (row.sj_pos - offset).astype(np.int32)
    inter_bb = Bbox.intersection(input_bb, Bbox(pos - rad, pos + rad))
    if np.product(inter_bb.size3()) == 0: 
      continue
    local_slc = inter_bb.to_slices()
    local_offset = offset + inter_bb.minpt
    
    local_sj_labels = sj_labels[local_slc]
    local_sj_mask = local_sj_labels == row.sj_id
    local_seg_chunk = seg_chunk[local_slc]
    
    post_seg_entries = get_neighbors(local_sj_mask, local_seg_chunk, 
      border_thickness=border_thickness, min_size=5)
    pre_seg_entry = [ps for ps in post_seg_entries if ps['id'] == row.pre_seg_id]
    # print(len(pre_seg_entry))
    if len(pre_seg_entry) < 1:
      # if cannot find pre_seg_id in neighbor, which is rare, use vc_pos as pre pos
      pre_pos = np.array(row.vc_pos)
    else:
      # if can find pre_seg_id in neighbor, use the median as pre seg pos
      pre_pos = pre_seg_entry[0]['pos'] + local_offset
    post_seg_entries = [ps for ps in post_seg_entries if ps['id'] != row.pre_seg_id and ps['id'] not in pre_seg_ids]

    
    # chose the largest one that forms obtuse angle vc - sj - post
    for ps in post_seg_entries:
      ps['angle'] = get_angle(pre_pos, row.sj_pos, ps['pos'] +local_offset) 
      # ps['angle'] = get_angle(row.vc_pos, row.sj_pos, ps['pos'] +local_offset) 
    # if row.pre_seg_id == 22525544:
    # if row.pre_seg_id == 22524202:
    # if row.pre_seg_id == 22522701:
    # if row.pre_seg_id == 22521966:
    # if row.sj_id == 247:
    #   pprint(post_seg_entries)
  
    #   print(row.sj_id)
    #   pprint(post_seg_entries)
    post_seg_entries = [ps for ps in post_seg_entries if ps['angle'] > max_angle]
    if not post_seg_entries: continue
    # each sj can only have one post syn seg partner
    max_ps = max(post_seg_entries, key=lambda ps: ps['size'])
    post_pos = max_ps['pos'] + local_offset 

    synapse_entries.append({
      'pre_seg_id': row.pre_seg_id,
      'post_seg_id': max_ps['id'],
      'vc_id': row.vc_id,
      'vc_pos': row.vc_pos,
      'vc_size': row.vc_size,
      'sj_id': row.sj_id,
      'sj_pos': row.sj_pos.tolist(),
      'sj_size': row.sj_size,
      'sj_norm_size': row.sj_norm_size,
      'sj_value': row.sj_value,
      'pre_seg_pos': pre_pos.tolist(),
      'post_seg_pos': post_pos.tolist()
    })
  new_synapse_df = pd.DataFrame(synapse_entries)
  line_annos = [neuroglancer.LineAnnotation(
    point_a=row.vc_pos,
    point_b=row.post_seg_pos,
    id=ind) for ind, row in new_synapse_df.iterrows()]
  return new_synapse_df, line_annos

def analyze_synapse(
  segmentation_vol, vc_vol, sj_vol, mask_vol,
  output_dir, chunk_size, overlap, offset, size):

  vc_thresh = 5
  sj_thresh = 5
  chunk_size = np.array(chunk_size)
  overlap = np.array(overlap)
  if mpi_rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    cv_args = dict(mip=0, progress=False, parallel=False, fill_missing=True, bounded=False)
    seg_cv = CloudVolume('file://%s' % segmentation_vol, **cv_args)
    vc_cv = CloudVolume('file://%s' % vc_vol, **cv_args)
    sj_cv = CloudVolume('file://%s' % sj_vol, **cv_args)
    mask_cv = CloudVolume('file://%s' % mask_vol, **cv_args)
    # union_bb = Bbox.intersection(seg_cv.meta.bounds(0), vc_cv.meta.bounds(0))
    # center = np.array([4349, 4435, 963])
    # size = np.array([512, 512, 128])
    # offset = center - size // 2
    if offset is None or size is None:
      union_bb = Bbox.intersection(seg_cv.meta.bounds(0), vc_cv.meta.bounds(0))
      offset = union_bb.minpt
      size = union_bb.size3()
    offset = np.array(offset)
    size = np.array(size)
    print(offset, size)

    union_bb = Bbox(offset, offset + size)
    print(union_bb)
    bbs = get_chunk_bboxes(union_bb, chunk_size, overlap)
    print(len(bbs))
    sub_bbs = np.array_split(bbs, mpi_size)
  else:
    seg_cv = None
    vc_cv = None
    sj_cv = None
    mask_cv = None
    sub_bbs = None

  seg_cv = mpi_comm.bcast(seg_cv, 0)
  vc_cv = mpi_comm.bcast(vc_cv, 0)
  sj_cv = mpi_comm.bcast(sj_cv, 0)
  mask_cv = mpi_comm.bcast(mask_cv, 0)
  sub_bbs = mpi_comm.scatter(sub_bbs, 0)
  
  padding = overlap // 2
  all_vc_dfs = []
  all_syn_dfs = []
  for ind, bb in tqdm(enumerate(sub_bbs), total=len(sub_bbs), desc='iterate bbs'):
    bb = Bbox(bb.minpt + padding, bb.maxpt - padding)
    offset = bb.minpt
    seg_chunk = np.array(seg_cv[bb])[..., 0]
    vc_chunk = np.array(vc_cv[bb])[..., 0]
    sj_chunk = np.array(sj_cv[bb])[..., 0]
    mask_chunk = np.array(mask_cv[bb])[..., 0]
    if np.logical_or.reduce(seg_chunk.ravel()) == 0:
      # logging.warning('empty seg')
      continue


    vc_df, vc_labels = find_vc(
      seg_chunk, mask_chunk, vc_chunk, offset, 
      vc_thresh=vc_thresh, size_thresh=100)
    if vc_df is None:
      # logging.warning('no vc')
      continue
    all_vc_dfs.append(vc_df)

    pre_synapse_df, sj_labels  = find_sj(
      vc_df, seg_chunk, vc_labels, mask_chunk, sj_chunk, offset, 
      pad=(3, 3, 2), border_thickness=(3, 3, 2), min_sj_size=60, 
      max_neighbor_count=3)
    if pre_synapse_df is None:
      # logging.warning('no sj')
      continue
    synapse_df, sj_psd_annos = find_post_syn(pre_synapse_df, seg_chunk, sj_labels, 
      offset, rad=(20, 20, 3), max_angle=60.0, border_thickness=(3, 3, 2))


    if len(synapse_df):
      cube_df_path = os.path.join(output_dir, 'synapse_%d_%d_%d.csv' % (offset[0], offset[1], offset[2]))
      synapse_df = synapse_df.set_index(['pre_seg_id', 'post_seg_id'])
      synapse_df.to_csv(cube_df_path)

      all_syn_dfs.append(synapse_df)
    # if ind > 10:
    #   break

  mpi_comm.barrier()
  all_vc_dfs = mpi_comm.reduce(all_vc_dfs, MPI.SUM, 0)
  all_syn_dfs = mpi_comm.reduce(all_syn_dfs, MPI.SUM, 0)
  if mpi_rank == 0:
    # print(all_dfs)
    # all_dfs = []
    all_vc_df = pd.concat(all_vc_dfs)
    vc_out_path = os.path.join(output_dir, 'vc.csv')
    all_vc_df.to_csv(vc_out_path)

    all_syn_df = pd.concat(all_syn_dfs)
    syn_out_path = os.path.join(output_dir, 'synapse.csv')
    all_syn_df.to_csv(syn_out_path)
  


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("segmentation_vol", help="segmentation volume")
  parser.add_argument("vc_vol", help="vc volume")
  parser.add_argument("sj_vol", help="sj volume")
  parser.add_argument("--mask_vol", help="mask volume")
  parser.add_argument("--output_dir", default=None)
  parser.add_argument("--chunk_size", default='512,512,128')
  parser.add_argument("--overlap", default='32,32,16')
  parser.add_argument("--offset", default=None)
  parser.add_argument("--size", default=None)

  # params to control syn finding

  args = parser.parse_args()
  chunk_size = [int(i) for i in args.chunk_size.split(',')]
  overlap = [int(i) for i in args.overlap.split(',')]
  if args.offset:
    offset = [int(i) for i in args.offset.split(',')]
  else:
    offset = None
  
  if args.size:
    size = [int(i) for i in args.size.split(',')]
  else:
    size = None

  analyze_synapse(args.segmentation_vol, args.vc_vol, args.sj_vol, args.mask_vol, 
    args.output_dir, chunk_size, overlap, offset, size)

if __name__ == "__main__":
  main()