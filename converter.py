import numpy as np
from scipy.misc import imsave
import h5py

for i in range(1, 157):
    # update user
    print "Converting image #:%03d" % i

    # read in image
    f = h5py.File('/mnt/win/Documents/FixedHSI/Image_%d.mat' % i)
    I = f['img'][:]
    
    # extract color bands
    R = I[25, :, :]
    G = I[15, :, :]
    B = I[5, :, :]
    
    # scale
    scale = 255/0.025
    R *= scale
    G *= scale
    B *= scale
    
    # clip
    R[R<0] = 0; R[R>255] = 255
    G[G<0] = 0; G[G>255] = 255
    B[B<0] = 0; B[B>255] = 255
    
    # quantize
    R = R.astype('uint8')
    G = G.astype('uint8')
    B = B.astype('uint8')
    
    # form output image
    I = np.zeros((1500, 1500, 3), dtype='uint8')
    I[:, :, 0] = R
    I[:, :, 1] = G
    I[:, :, 2] = B
    
    # save image
    imsave('/mnt/win/Documents/rgb/image%03d.jpg' % i, I)
