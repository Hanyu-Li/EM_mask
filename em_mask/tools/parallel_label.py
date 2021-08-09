import argparse
import os
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from ffn.utils import bounding_box
from em_mask.precomputed_utils import ffn_to_cv, get_chunk_bboxes, prepare_precomputed
from tqdm import tqdm
import logging
from scipy import ndimage
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def label_volume(input, output, sub_box_size, overlap, verbose=False):
  '''Remap input precomputed to output chunkwise.'''
  in_cv = CloudVolume(f'file://{input}', mip=0, parallel=False, bounded=False, fill_missing=True)
  if mpi_rank == 0:
    in_bounds = in_cv.bounds
    in_offset = in_bounds.minpt
    in_size = in_bounds.maxpt - in_bounds.minpt
    resolution = in_cv.resolution

    logging.warning('In bounds %s', in_bounds)

    bbs = get_chunk_bboxes(
      in_bounds, 
      chunk_size=sub_box_size, 
      overlap=overlap, 
      include_small_sub_boxes=False)

    bbs = np.array_split(bbs, mpi_size)
    
  else:
    bbs = None
    resolution = None
  
  bbs = mpi_comm.scatter(bbs, 0)
  resolution = mpi_comm.bcast(resolution, 0)

  for bb in tqdm(bbs, desc='Iterate bbox', disable=not verbose):

    out_name = 'precomputed-%d_%d_%d_%d_%d_%d' % (
      bb.minpt[0], bb.minpt[1], bb.minpt[2], 
      bb.size3()[0], bb.size3()[1], bb.size3()[2]) 

    segmentation_output_dir = os.path.join(output, out_name)
    os.makedirs(segmentation_output_dir, exist_ok=True)
    # out_path = os.path.join(output, )
    out_cv = prepare_precomputed(segmentation_output_dir, bb.minpt, bb.size3(), 
      resolution, chunk_size=(128, 128, 64), layer_type='segmentation')
    in_chunk = in_cv[bb][..., 0]
    # out_chunk = np.stack([
      # clahe_image(in_chunk[:, :, z]) for z in range(in_chunk.shape[2])], axis=2)
    # out_chunk[mask_chunk] = 0
    out_chunk = ndimage.label(in_chunk)[0].astype(np.uint32)
    out_cv[bb] = out_chunk

def parse_args():
  """Parse command line arguments."""
  p = argparse.ArgumentParser()

  p.add_argument('input', type=str,
                  help='path to input precomputed')
  p.add_argument('output', type=str,
                  help='path to output precomputed')
  p.add_argument('--sub_box_size', type=str, default='1024,1024,256', help='sub box size')
  p.add_argument('--overlap', type=str, default='64,64,32', help='sub box size')
  p.add_argument('--verbose', action='store_true')
  return p.parse_args()

def main():
    """Command line entry point to convert images to a CloudVolume layer."""
    args = parse_args()
    sub_box_size = [int(i) for i in args.sub_box_size.split(',')]
    overlap = [int(i) for i in args.overlap.split(',')]
    label_volume(args.input, args.output, sub_box_size, overlap, args.verbose)


if __name__ == '__main__':
    main()