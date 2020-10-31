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

def merge_dict(a, b, datatype):
  a.update(b)
  return a

def count_length(segmentation_vol, id_text, output_dir):
  if mpi_rank == 0:
    ids = np.loadtxt(id_text)
    print(len(ids))
    sub_ids = np.array_split(ids, mpi_size)
  else:
    sub_ids = None

  seg_cv = CloudVolume('file://%s' % segmentation_vol, mip=1)
  # sks = seg_cv.skeleton

  sub_ids = mpi_comm.scatter(sub_ids, 0)

  sk_len_dict = {}
  for i in tqdm(sub_ids[:]):
    try:
      sk = seg_cv.skeleton.get(int(i))
      length = sk.cable_length()
      sk_len_dict[i] = length / 1000.0
    except:
      pass
  
  mergeOp = MPI.Op.Create(merge_dict, commute=True)
  sk_len_dict = mpi_comm.reduce(sk_len_dict, op=mergeOp, root=0)
  if mpi_rank == 0:
    # pprint(sk_len_dict)
    print('total_lenth', np.sum(list(sk_len_dict.values())))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'len_dict.pkl'), 'wb') as f:
      pickle.dump(sk_len_dict, f)



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("segmentation_vol", help="segmentation volume")
  parser.add_argument("id_text", help="mask volume")
  parser.add_argument("--output_dir", default=None)
  args = parser.parse_args()

  count_length(args.segmentation_vol, args.id_text, args.output_dir)

if __name__ == '__main__':
  main()