#Task 5: Gradient domain image enhancement
import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy import signal
import skimage
from scipy.interpolate import interp1d
from skimage import color

from task4_grad_domain import img2grad_field, reconstruct_grad_field

if __name__ == "__main__":
    #TODO: Replace with your own image
    im = io.imread(path.join('images','task5b-1.jpg'))
    im = skimage.img_as_float(im)

    im_gray = color.rgb2gray(im)
    
    G = img2grad_field(im_gray)

    #TODO: Implement gradient domain enhancement on the greyscale image, then recover colour
    #Hint: Use reconstruct_grad_field from the previous task

    #To brighten up images, use m < 1 e.g.1.5
    #To increase image contrast, use m > 1 e.g.0.75
    x_in = 0.4
    m = 2

    x = np.arange(-2.0, 2.0, 0.1)
    y = m*x
    f1 = interp1d(x, y)

    m2 = (m*x_in -1)/(x_in - 1)
    y2 = m2*x + (1-m2)
    f2 = interp1d(x, y2)

    G_enhanced = np.zeros(G.shape)
    G_enhanced = np.where(G < x_in, f1(G), f2(G))

    print("No. of pixels: ", G.shape[0]*G.shape[1])

    Gm = np.sqrt(np.sum(G_enhanced*G_enhanced, axis=2))
    w = 1/(Gm + 0.0001)     # to avoid pinching artefacts
    imr = reconstruct_grad_field(G_enhanced,w,im_gray[0,0], im_gray, "cholesky").clip(0,1)
    # imr = reconstruct_grad_field(G_enhanced,w,im_gray[0,0], im_gray, "sp").clip(0,1)


    imr_col = im.copy()
    c = imr/(im_gray+ 0.0001) # 0.0001 to avoid diversion by 0
    for i in range(0,3):
        imr_col[:,:,i] = im[:,:,i]*c


    plt.figure(figsize=(9, 3))

    plt.subplot(121)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(im)

    plt.subplot(122)
    plt.title('Enhanced')
    plt.axis('off')
    plt.imshow(imr_col)

    plt.show()
    # plt.savefig('./results/task5b.png')

    # io.imsave(path.join('results','gd_enhanced.jpg'), imr_color)
