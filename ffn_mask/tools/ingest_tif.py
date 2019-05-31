
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tifffile
import h5py
import numpy as np
from glob import glob
import skimage
from absl import app
from absl import flags
import sys
import re
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

FLAGS = flags.FLAGS

flags.DEFINE_string('tif_dir', None, '')
flags.DEFINE_string('h5_output', None, '')
flags.DEFINE_integer('z_axis', 0, '')
flags.DEFINE_boolean('clahe', False, '')

index_match = re.compile(r'.*/.*_([0-9]+)\.(.*)')
def get_ind(filename_str):
  return int(index_match.match(filename_str).group(1))
def main(unused_argv):
  tif_list = glob(FLAGS.tif_dir+'/*.tif*')
  tif_list.sort()
  tif_lists = np.array_split(tif_list, comm.size)

  path, dataset = FLAGS.h5_output.split(':')

  f = h5py.File(path, 'w', driver='mpio', comm=MPI.COMM_WORLD)
  sample_im = tifffile.imread(tif_list[0])
  z = len(tif_list)
  y,x = sample_im.shape
  volume = f.create_dataset(dataset, shape=(z,y,x), dtype=np.uint8)
  for f_tif in tif_lists[rank]:
    i = get_ind(f_tif)
    print(i, f_tif)
    input_image = tifffile.imread(f_tif)
    filtered_image = skimage.exposure.equalize_adapthist(
      image=input_image, kernel_size=256)
    volume[i,:,:] = np.round(filtered_image * 255).astype(np.uint8) 
    # if i == 2:
    #   break
    # volume[i,:,:]

  f.close()




if __name__ == '__main__':
  app.run(main)

