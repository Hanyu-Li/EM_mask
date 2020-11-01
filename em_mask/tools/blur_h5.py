import logging
import cloudvolume
from cloudvolume.lib import Bbox
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from tqdm import tqdm
import h5py
import os
import glob
import re
import neuroglancer
import h5py
import skimage
from scipy import ndimage
import argparse

def blur_volume(input_h5, output_h5, sigma=2, z_step=32):
  with h5py.File(input_h5, 'r') as f_in:
    with h5py.File(output_h5, 'w') as f_out:
      lg = f_in['logits']
      shape = lg.shape[:-1]
      print(shape)
      z_sets = np.arange(0, lg.shape[0], z_step)
      ds_out = f_out.create_dataset('class_prediction', shape=shape, dtype=np.uint8)
      pad = 2
      for z in tqdm(z_sets):
        r_z_start = z - pad
        w_z_start = z
        if r_z_start < 0:  r_z_start = 0
        r_z_end = z + z_step + pad
        w_z_end = z + z_step
        if r_z_end >= shape[0]:  r_z_end = shape[0] - 1
        if w_z_end >= shape[0]:  w_z_end = shape[0] - 1
        chunk = lg[r_z_start:r_z_end, :, :, 0]
        blur_chunk = ndimage.filters.gaussian_filter(chunk, sigma=sigma, mode='reflect')
        
        sel_start = w_z_start - r_z_start 
        sel_end = blur_chunk.shape[0] - (r_z_end - w_z_end)
        logging.warning('r: %s w: %s sel: %s', (r_z_start, r_z_end), (w_z_start, w_z_end), (sel_start, sel_end))
        
        select_blur_chunk = blur_chunk[sel_start:sel_end, ]
        logging.warning('padded shape %s', select_blur_chunk.shape)
        ds_out[w_z_start:w_z_end, :, :] = np.uint8(select_blur_chunk > 0)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_h5", help="Input h5")
  parser.add_argument("output_h5", help="Output h5")
  parser.add_argument("-s", "--sigma", default=2)
  args = parser.parse_args()
  blur_volume(args.input_h5, args.output_h5, args.sigma)

if __name__ == "__main__":
  main()