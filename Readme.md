
The virtual environment is included (Folder: ve_practical1). All the images are in the image folder and labelled with the task number. Most of the files take some time to run, espectially with higher resolution image. Can change the image name in the code files to run images correspond to that task. 
The results from various tasks have been documented in the result folder as well.

## Task 2 special remarks:
Can try a different alpha mask, just have to uncomment:
```
  a4 = np.ones(int(0.5*w - w*window_size*0.5*0.5))
  a5 = np.linspace(1, 0, int(w*window_size*0.5))
  a6 = np.zeros(w-a4.shape[0]-a5.shape[0])
  alpha_sharp = np.concatenate((a4, a5, a6), axis=None)

  alp1 = np.tile(alpha,(int(h/4),1))
  alp2 = np.tile(alpha_sharp,(int(h/2),1))
  alp3 = np.tile(alpha,((h-int(h/4)-int(h/2)),1))
  alpha_mask = np.concatenate((alp1,alp2,alp3)) 
```

## Task 5 special remarks:
Can change the slope of the gradient enhancement function to obtain a brightening/ boost contrast results

## Task 6 special remarks:
#### a) After running the code for the first time, can comment out
```
    Ox, Oy = grad_operator(mask, bg, pixel_actual_row, pixel_actual_col)
    sparse.save_npz("cacheOx.npz", Ox)
    sparse.save_npz("cacheOy.npz", Oy)
```
and uncomvment 
```
    Ox = sparse.load_npz("cacheOx.npz")
    Oy = sparse.load_npz("cacheOy.npz")
```
As partial derivatives operator are expensive to compute, the results have been cached and can be reloaded if the same image is being processed.
#### b) Can switch between "Importing gradient" and "Mixing gradient" methods in the function "reconstruct_grad_field"