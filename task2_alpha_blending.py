#Task 2: Alpha blending

import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util
from skimage.transform import pyramid_gaussian

import matplotlib.pyplot as plt

def hard_blending(im_left, im_right):
  """
  return an image that consist of the left-half of im_left
  and right-half of im_right
  """
  assert(im_right.shape == im_left.shape)
  h, w, c = im_right.shape
  new_im = im_right.copy()
  new_im[:,:(w//2),:] = im_left[:,:(w//2),:]
  return new_im

def alpha_blending(im_left, im_right, window_size=0.5):
  """
  return a new image that smoothly combines im1 and im2
  im_left: np.array image of the dimensions: height x width x channels; values: 0-1 
  im_right: np.array same dim as im_left
  window_size: what fraction of image width to use for the transition (0-1)
  """
  # useful functions: np.linspace and np.concatenate
  assert(im_right.shape == im_left.shape)
  # TODO: Put your code below
  im_left = im_left.reshape(im_left.shape[0],im_left.shape[1],-1) # Add an extra dimension if grayscale
  im_right = im_right .reshape(im_right.shape[0],im_right.shape[1],-1) # Add an extra dimension if grayscale

  h, w, c = im_left.shape
  #Alpha mask for left hand side image
  a1 = np.ones(int(0.5*w) - int(w*window_size*0.5))
  a2 = np.linspace(1, 0, int(w*window_size))
  a3 = np.zeros(w-a1.shape[0]-a2.shape[0])
  alpha = np.concatenate((a1, a2, a3), axis=None)
  alpha_mask = np.tile(alpha,(h,1))

  #Try to apply gradual change vertically as well
  # a4 = np.ones(int(0.5*w - w*window_size*0.5*0.5))
  # a5 = np.linspace(1, 0, int(w*window_size*0.5))
  # a6 = np.zeros(w-a4.shape[0]-a5.shape[0])
  # alpha_sharp = np.concatenate((a4, a5, a6), axis=None)

  # alp1 = np.tile(alpha,(int(h/4),1))
  # alp2 = np.tile(alpha_sharp,(int(h/2),1))
  # alp3 = np.tile(alpha,((h-int(h/4)-int(h/2)),1))
  # alpha_mask = np.concatenate((alp1,alp2,alp3)) 

  new_img = im_left.copy()
  
  for i in range(0,im_left.shape[2]):
      new_img[:,:,i] = alpha_mask*im_left[:,:,i] + (1-alpha_mask)*im_right[:,:,i]
  return new_img
  

if __name__ == "__main__":
  # TODO: Replace with your own images
  im_left = io.imread(path.join('images','task2-1-left.jpg'))
  # im_left = io.imread(path.join('images','task2-2-left.jpg'))
  im_left = util.img_as_float(im_left[:,:,:3])
  im_right = io.imread(path.join('images','task2-1-right.jpg'))
  # im_right = io.imread(path.join('images','task2-2-right.jpg'))
  im_right = util.img_as_float(im_right[:,:,:3])
  plt.figure(figsize=(20, 16))

  plt.subplot(221)
  plt.title('left image')
  plt.axis('off')
  plt.imshow(im_left)

  plt.subplot(222)
  plt.title('right image')
  plt.axis('off')
  plt.imshow(im_right)

  plt.subplot(223)
  plt.title('hard blending')
  plt.axis('off')
  plt.imshow(hard_blending(im_left, im_right))

  plt.subplot(224)
  plt.title('alpha blending')
  plt.axis('off')
  plt.imshow(alpha_blending(im_left, im_right, window_size=0.25))

  plt.savefig('./results/task2.png')

  plt.show()
