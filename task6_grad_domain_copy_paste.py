#Task 6: Gradient domain copy & paste

import os.path as path
import skimage.io as io
import numpy as np
from skimage.util import img_as_uint
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
from scipy import ndimage
from sksparse.cholmod import cholesky
import time

def grad_operator(mask, img, pixel_actual_row, pixel_actual_col):
    """
    Return the Gradient operators as sparse matrices
    """
    N = pixel_actual_col.shape[0]

    Ox_row = np.zeros(N*2)
    Ox_col = np.zeros(N*2)
    Ox_data = np.zeros(N*2)
    Oy_row = np.zeros(N*2)
    Oy_col = np.zeros(N*2)
    Oy_data = np.zeros(N*2)
    index_x = 0
    index_y = 0
    for n in range(N):
        x = pixel_actual_row[n]
        y = pixel_actual_col[n]

        if (mask[x+1, y] != 0):
            Ox_row[index_x] = n
            Ox_col[index_x] = n
            Ox_data[index_x] = -1
            index_x += 1

            possible_i1 =  np.where(pixel_actual_row == (x+1))
            possible_i2 = np.where(pixel_actual_col == y)
            i = np.intersect1d(possible_i1, possible_i2)[0]

            Ox_row[index_x] = n
            Ox_col[index_x] = i
            Ox_data[index_x] = 1
            index_x += 1

        if (mask[x, y+1] != 0):
            Oy_row[index_y] = n
            Oy_col[index_y] = n
            Oy_data[index_y] = -1
            index_y += 1

            possible_j1 =  np.where(pixel_actual_row == x)
            possible_j2 = np.where(pixel_actual_col == (y+1))
            j = np.intersect1d(possible_j1, possible_j2)[0]

            Oy_row[index_y] = n
            Oy_col[index_y] = j
            Oy_data[index_y] = 1
            index_y += 1


    Ox = sparse.coo_matrix((Ox_data, (Ox_row, Ox_col)), shape=(N, N)).asformat(format = 'csr')
    Oy = sparse.coo_matrix((Oy_data, (Oy_row, Oy_col)), shape=(N, N)).asformat(format = 'csr')

    print("Finish calculating gradient operator")

    return Ox, Oy

def img2grad(mask, b, f, N):
    """Return a gradient field for the region specified by the mask
    The function returns image [height,width,2], where the last dimension selects partial derivates along x or y
    """
    maskt = mask.T
    non_zero_row = np.nonzero(maskt)[1]
    non_zero_col = np.nonzero(maskt)[0]

    grad = np.zeros((1,N,2))

    for i in range(N):
        next_x = non_zero_row[i] + 1
        next_y = non_zero_col[i] + 1
        x = non_zero_row[i]
        y = non_zero_col[i]

        #Use normal gradient techniques
        # grad[0,i,0] = f[next_x,y] - f[x, y] #gradx of background
        # grad[0,i,1] = f[x,next_y] - f[x, y] #grady

        #Using mixing gradient techniques
        gradfx = f[next_x,y] - f[x, y] #grady of background
        gradfy = f[x,next_y] - f[x, y] #gradx
        gradbx = b[next_x,y] - b[x, y] #grady of background
        gradby = b[x,next_y] - b[x, y] #gradx
        if (abs(gradfx)) > (abs(gradbx)):
            grad[0, i, 0] = gradfx
        else:
            grad[0, i, 0] = gradbx

        if (abs(gradfy)) > (abs(gradby)):
            grad[0, i, 1] = gradfy
        else:
            grad[0, i, 1] = gradby


    return grad


def reconstruct_grad_field(Ox, Oy, Gf, Et, T):

    Oxt = Ox.transpose()
    Oyt = Oy.transpose()

    Gx_ini = Gf[:,:,0].flatten(order='F').reshape(-1,1) 
    Gx = sp.sparse.csr_matrix(Gx_ini)
    Gy_ini = Gf[:,:,1].flatten(order='F').reshape(-1,1) 
    Gy = sp.sparse.csr_matrix(Gy_ini)

    Et = E.transpose() # N by K
    A = Oxt @ Ox + Oyt @ Oy + Et @ E
    b = Oxt @ Gx + Oyt @ Gy + Et @ T

    print("finish computing Ab ", A.shape, b.shape)

    #Use Cholesky solver
    factor = cholesky(A)
    x = factor(b)

    return x.toarray()


if __name__ == "__main__":

    # Read background image
    bg = io.imread(path.join('images','task6-1.jpg'))[:,:,:3]
    bg = skimage.img_as_float(bg)
    # Read foreground image
    fg = io.imread(path.join('images','task6-2.png'))
    fg = skimage.img_as_float(fg)
    # Calculate alpha mask
    mask = (fg[:,:,3] > 0.5).astype(int)
    fg = fg[:,:,:3] # drop alpha channel

    #TODO: Implement gradient-domain copy&paste. 

    #calculate gradient operator
    flat_mask = mask.flatten(order = 'F')
    maskt = mask.T
    pixel_actual_row = np.nonzero(maskt)[1] #becuase np.non zero is row major not column major, have to transpose it, then swap the column and row indexes
    pixel_actual_col = np.nonzero(maskt)[0]

    N = pixel_actual_row.shape[0]
    Ox, Oy = grad_operator(mask, bg, pixel_actual_row, pixel_actual_col)
    sparse.save_npz("cacheOx.npz", Ox)
    sparse.save_npz("cacheOy.npz", Oy)
    # Ox = sparse.load_npz("cacheOx.npz") #load pre-calculated gradient if re-processing the same image
    # Oy = sparse.load_npz("cacheOy.npz") #load pre-calculated gradient if re-processing the same image

    #calculate whether that pixel belong to an edge
    ero_mask = ndimage.binary_erosion(mask)
    edge = mask - ero_mask
    edget = edge.T

    edge_actual_row = np.nonzero(edget)[1]
    edge_actual_col = np.nonzero(edget)[0]
    K = edge_actual_row.shape[0]


    x = np.zeros((N, bg.shape[2]))
    I_dest = bg.copy()
    for cc in range(bg.shape[2]):
        print("starting colour channel: ", cc)

        #calculate gradient of the region to be pasted, and pasted region
        b = bg[:,:,cc]
        bm = bg[:,:,cc]*mask
        f = fg[:,:,cc]
        fm = fg[:,:,cc]*mask
        G_f = img2grad(mask, b, f, N)
        # G = img2grad(mask, f, b, N) #for debug, paste background gradient on background image

        #formulate E matrix with edge as the diagonals
        #formulate T vector with edge values
        edge_diag = np.zeros(N)
        I = np.zeros(N) #background cropped region, unrolled
        T = np.zeros(K)
        k = 0
        for n in range(N):
            I[n] = b[pixel_actual_row[n], pixel_actual_col[n]].clip(0,1)
            if (pixel_actual_row[n] == edge_actual_row[k]) and (pixel_actual_col[n] == edge_actual_col[k]):
                edge_diag[n] = 1
                T[k] = b[pixel_actual_row[n], pixel_actual_col[n]]
                k += 1
            else:
                edge_diag[n] = 0

        E = sparse.spdiags(edge_diag,0, K, N, format = 'csr') # K by N
        T = sp.sparse.csr_matrix(T.reshape(-1,1)) # K by 1
  
        x[:,cc] = reconstruct_grad_field(Ox, Oy, G_f, E, T).flatten(order = 'F').clip(0,1)

    for n in range(N):
        I_dest[pixel_actual_row[n], pixel_actual_col[n], 0] = x[n, 0]
        I_dest[pixel_actual_row[n], pixel_actual_col[n], 1] = x[n, 1] 
        I_dest[pixel_actual_row[n], pixel_actual_col[n], 2] = x[n, 2]


    # Naive copy-paste for comparision
    mask3 = np.reshape(mask,[mask.shape[0], mask.shape[1], 1]) 
    I_naive = fg*mask3 + bg*(1-mask3)
    
    plt.figure(figsize=(9, 9))
    plt.subplot(121)
    plt.title('Naive')
    plt.axis('off')
    io.imshow(I_naive)

    plt.subplot(122)
    plt.title('Poisson Blending')
    plt.axis('off')
    io.imshow(I_dest)

    plt.show()

    # io.imsave(path.join('results','copy_paste.jpg'), skimage.img_as_ubyte(I_dest))
