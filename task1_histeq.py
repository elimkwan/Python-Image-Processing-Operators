# Task 1: Image enhancement
import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util

import matplotlib.pyplot as plt

img_name = 'task1b-3.jpg'

#--------------------------------------------------------------
#------------------------Part 1--------------------------------
#--------------------------------------------------------------

def equalise_hist(image, bin_count=256):
  """
  Perform histogram equalization on an image and return as a new image.

  Arguments:
  image -- a numpy array of shape height x width, dtype float, range between 0 and 1
  bin_size -- how many bins to use
  """
  # TODO: your histogram equalization code
  new_image = image.copy()
  flat = new_image.flatten()

  #Generate histogram
  quant = lambda t: round(t * 255) 
  quantization = np.vectorize(quant) 
  quant = quantization(flat)
  hist = np.bincount(quant)

  #Generate cdf
  cdf = np.cumsum(hist)
  a = cdf.min() 
  b = cdf.max()
  cdf_norm = (cdf-a)/(b-a)

  #Use cdf as look up table
  new_px = np.empty(quant.shape)
  count = 0
  for i in quant:
    new_px[count] = cdf_norm[i]
    count += 1

  new_image[:, :] = new_px.reshape(new_image.shape)

  return new_image.clip(0,1)


# TODO: Change the file to your own image

test_im = io.imread(path.join('images',img_name))
test_im_gray = color.rgb2gray(test_im)
fig1 = plt.figure(figsize=(12, 15))
plt.subplot(121)
plt.title('Original image')
plt.axis('off')
plt.imshow(test_im_gray,cmap='gray')

plt.subplot(122)
plt.title('Histogram equalised image')
plt.axis('off')
plt.imshow(equalise_hist(test_im_gray),cmap='gray')

plt.show()
# fig1.savefig('./results/task1-1.png')
# plt.close(fig1)

#--------------------------------------------------------------
#------------------------Part 2--------------------------------
#--------------------------------------------------------------

def he_per_channel(img):
  # Perform histogram equalization separately on each colour channel. 
  # TODO: put your code below
  new_img = img.copy()
  for i in range(0,3):
    new_img[:, :, i] = equalise_hist(img[:, :, i])

  return new_img.clip(0,1)
  

def he_colour_ratio(img):
  # Perform histogram equalization on a gray-scale image and transfer colour using colour ratios.
  # TODO: put your code below
  new_img = img.copy()
  v_old = color.rgb2gray(img)
  v_new = equalise_hist(v_old)
  c = v_new/v_old
  for i in range(0,3):
      new_img[:,:,i] = img[:,:,i]*c

  return new_img.clip(0,1)
  

def he_hsv(img):
  # Perform histogram equalization by processing channel V in the HSV colourspace.
  # TODO: put your code below
  hsv_img = color.rgb2hsv(img)
  value_img = hsv_img[:,:,2]
  hsv_img[:,:,2] = equalise_hist(value_img)
  new_img = color.hsv2rgb(hsv_img)
  return new_img.clip(0,1)
  



test_im = io.imread(path.join('images',img_name))
test_im = util.img_as_float(test_im)
fig2 = plt.figure(figsize=(50, 30))

plt.subplot(121)
plt.title('Original image')
plt.axis('off')
io.imshow(test_im)


plt.subplot(122)
plt.title('Each channel processed seperately')
plt.axis('off')
io.imshow(he_per_channel(test_im))

plt.savefig('./results/task1-2-1.png')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
fig3 = plt.figure(figsize=(50, 30))

plt.subplot(121)
plt.title('Gray-scale + colour ratio')
plt.axis('off')
io.imshow(he_colour_ratio(test_im))


plt.subplot(122)
plt.title('Processed V in HSV')
plt.axis('off')
io.imshow(he_hsv(test_im))


# plt.savefig('./results/task1-2-2.png')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

