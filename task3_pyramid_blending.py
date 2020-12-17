#Task 3: Pyramid blending

import os.path as path
import skimage.io as io

import math
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt

from task2_alpha_blending import alpha_blending

def gausspyr_reduce(x, kernel_a=0.4):
  """
  Filter and subsample the image x. Used to create consecutive levels of the Gaussian pyramid [1]. Can process both grayscale and colour images. 
  x - image to subsample
  kernel_a - the coefficient of the kernel

  returns an image that is half the size of the input x. 

  [1] Burt, P., & Adelson, E. (1983). The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communications, 31(4), 532–540. https://doi.org/10.1109/TCOM.1983.1095851
  """

  #Kernel
  K = np.array( [ 0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2 ] )
  
  x = x.reshape(x.shape[0],x.shape[1],-1) # Add an extra dimension if grayscale
  y = np.zeros([math.ceil(x.shape[0]/2),math.ceil(x.shape[1]/2),x.shape[2]]) # Store the result in this array
  for cc in range(x.shape[2]): # for each colour channel
    #TODO: Add filtering and subsampling code
    # Step 1: filter rows
    # Step 2: subsample rows (skip every second column)
    # Step 3: filter columns
    # Step 4: subsample columns (skip every second row)

    y_filr = x[:,:,cc].copy()
    y_filr = sp.signal.convolve2d(x[:,:,cc], K.reshape(1,-1), boundary='symm', mode='same') #filter rows
    h, w = y_filr.shape 
    y_subr = np.zeros((h,int(w/2)))
    y_subr = y_filr[:,::2] #subsample rows
    y_filc = y_subr.copy()
    y_filc = sp.signal.convolve2d(y_subr, K.reshape(1,-1), boundary='symm', mode='same') #filter columns
    h, w = y_filc.shape 
    y[:,:,cc] = y_filc[::2,:] #subsample columns

  return np.squeeze(y) # remove an extra dimension for grayscale images
  
def gausspyr_expand(x, sz=None, kernel_a=0.4):
  """
  Double the size and interpolate using Gaussian pyramid kernel [1]. Can process both grayscale and colour images. 
  x - image to upsample
  sz - [height, width] of the generated image. Necessary if one of the dimensions of the upsampled image is odd. 
  kernel_a - the coefficient of the kernel

  returns an image that is  double the size or the size of sz of the input x. 

  [1] Burt, P., & Adelson, E. (1983). The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communications, 31(4), 532–540. https://doi.org/10.1109/TCOM.1983.1095851
  """

  # Kernel is multipled by 2 to preserve energy when increasing the resolution
  K = 2*np.array( [ 0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2 ] )


  if sz is None:
    sz = (x.shape[0]*2, x.shape[1]*2)
  
  x = x.reshape(x.shape[0],x.shape[1],-1) # Add an extra dimension if grayscale
  y = np.zeros([sz[0], sz[1], x.shape[2]])
  for cc in range(x.shape[2]): # for each colour channel
    y_a = np.zeros( (x.shape[0],sz[1]) )
    y_a[:,::2] = x[:,:,cc]
    y_a = sp.signal.convolve2d(y_a, K.reshape(1,-1), mode='same', boundary='symm') # filter rows
    y[::2,:,cc] = y_a
    y[:,:,cc] = sp.signal.convolve2d( y[:,:,cc], K.reshape(-1,1), mode='same', boundary='symm') # filter columns

  return np.squeeze(y) # remove an extra dimension for grayscale images


class laplacian_pyramid:

  @staticmethod
  def decompose( img, levels=-1 ):    
    """
    Decompose img into a Laplacian pyramid. 
    levels: how many levels should be created (including the base band). When the default (-1) value is used, the maximum possible number of levels is created. 
    """
    # The maximum number of levels we can have
    max_levels = math.floor(math.log2(min(img.shape[0], img.shape[1])))
    assert levels<max_levels
    if levels==-1: 
      levels = max_levels # Use max_levels by default
    pyramid = []

    #TODO: Implement Laplacian pyramid decomposition using gausspyr_reduce and gausspyr_expand
    next_img = np.zeros(img.shape)
    cur_img = img.copy()
    for i in range(0, levels-1):

      next_img = gausspyr_reduce(cur_img) #gasspyr reduction use math.ceil e.g. inital x = 5, guass_reduced x = (5+1)/2 = 3
      l = cur_img - gausspyr_expand(next_img, cur_img.shape)
      pyramid.append(l)
      cur_img = next_img

    pyramid.append(cur_img) #e.g. level 0:fine, level 5: coarse
    return pyramid

  @staticmethod
  def reconstruct( pyramid ):    
    """
    Combine the levels of the Laplacian pyramid to reconstruct an image. 
    """

    #TODO: Implement Laplacian pyramid reconstruction using gausspyr_expand
    levels = len(pyramid) - 1 #highest level

    cur_img = pyramid[levels].copy()
    for i in range(levels, 0, -1):
      expand_cur = gausspyr_expand(cur_img, pyramid[i-1].shape)
      next_img = expand_cur + pyramid[i-1]
      cur_img = next_img
    
    img = next_img

    return img


def pyramid_blending(im1, im2, levels=4, window_size=0.3):
  #TODO: Implement pyramid blending
  new_img = im1.copy()

  for j in range (0, im1.shape[2]):

    pyramid1 = laplacian_pyramid.decompose(im1[:,:,j], 4)
    pyramid2 = laplacian_pyramid.decompose(im2[:,:,j], 4)

    pyramid_total = []

    base_window_size = (window_size/(len(pyramid1)))
    for i in range (0, len(pyramid1)):
      #use wider window size for coarse levels, and narrower window for fine levels
      cur_window_size = base_window_size*(i+1)
      if (pyramid1[i].shape == pyramid2[i].shape):
        a = alpha_blending(pyramid1[i], pyramid2[i], cur_window_size)
        pyramid_total.append(np.squeeze(a))
        
    new_img[:,:,j] = laplacian_pyramid.reconstruct(pyramid_total)
  
  return new_img
  

if __name__ == "__main__":

  #Part 1: Laplacian pyramid decomposition
  #TODO: Replace with your own image
  im = io.imread(path.join('images','task3a.jpg'))
  im = util.img_as_float(im[:,:,:3])
  im = color.rgb2gray(im)

  pyramid = laplacian_pyramid.decompose(im, levels=4)

  plt.figure(figsize=(3*len(pyramid), 3))
  grid = len(pyramid) * 10 + 121
  
  for i, layer in enumerate(pyramid):
    plt.subplot(grid+i)
    plt.title('level {}'.format(i))
    plt.axis('off')
    if i == len(pyramid)-1:
      io.imshow(layer)
    else:
      plt.imshow(layer)
    
  plt.subplot(grid+len(pyramid))
  plt.title('reconstruction')
  plt.axis('off')
  im_reconstructed = laplacian_pyramid.reconstruct(pyramid)
  io.imshow(np.clip(im_reconstructed, 0, 1))

  plt.subplot(grid+len(pyramid)+1)
  plt.title('differences')
  plt.axis('off')
  plt.imshow(np.abs(im - im_reconstructed))

  plt.show()  
  # plt.savefig('./results/task3a.png')

  # Part 2: Pyramid blending
  # TODO: Replace with your own images
  im_left = io.imread(path.join('images','task3b-2-left.jpg'))
  im_left = util.img_as_float(im_left[:,:,:3])
  im_right = io.imread(path.join('images','task3b-2-right.jpg'))
  im_right = util.img_as_float(im_right[:,:,:3])

  fig2 = plt.figure(figsize=(15, 12))
  plt.subplot(221)
  plt.title('left image')
  plt.axis('off')
  plt.imshow(im_left)

  plt.subplot(222)
  plt.title('right image')
  plt.axis('off')
  plt.imshow(im_right)

  plt.subplot(223)
  plt.title('alpha blend')
  plt.axis('off')
  a = alpha_blending(im_left, im_right, window_size=0.02)
  plt.imshow(a)

  plt.subplot(224)
  plt.title('pyramid blend')
  plt.axis('off')
  p = pyramid_blending(im_left, im_right, window_size=0.02)
  plt.imshow(p)
  # plt.show()
  plt.savefig('./results/task3b.png')

  plt.title('differences')
  plt.axis('off')
  plt.imshow(np.abs(p - a))
  plt.show()
