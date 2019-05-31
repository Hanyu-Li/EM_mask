from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import h5py
import skimage
import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('h5_input', None, '')
flags.DEFINE_string('h5_output', None, '')
flags.DEFINE_integer('z_axis', 0, '')


def clahe(h5_input, h5_output):
  in_path, in_dataset = h5_input.split(':')
  input_volume = h5py.File(in_path,'r')[in_dataset]

  out_path, out_dataset = h5_output.split(':')
  f = h5py.File(out_path, 'w')
  output_volume = f.create_dataset(out_dataset, shape=input_volume.shape, 
    dtype=np.uint8)

  z = FLAGS.z_axis
  max_z = input_volume.shape[z]
  for i in range(max_z):
    if z == 0:
      filtered_image = skimage.exposure.equalize_adapthist(
        image=input_volume[i,:,:], kernel_size=256)
      output_volume[i,:,:] = np.round(filtered_image * 255).astype(np.uint8) 
    elif z == -1 or z == 2:
      filtered_image = skimage.exposure.equalize_adapthist(
        image=input_volume[:,:,i], kernel_size=256)
      output_volume[:,:, i] = np.round(filtered_image * 255).astype(np.uint8) 


def main(unused_argv):
  clahe(FLAGS.h5_input, FLAGS.h5_output)

if __name__ == '__main__':
  app.run(main)

  # input_image_filtered = map(lambda x: skimage.exposure.equalize_adapthist(image=x, kernel_size=256), input_image)