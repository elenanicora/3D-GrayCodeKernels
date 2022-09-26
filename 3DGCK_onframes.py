#!/usr/bin/env python
import binary_tree
import efficient_scheme

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import cv2
import scipy.signal
import time
import glob

### Function that resize the original video frames by a certain percentage and adjust the range of values of the images to be in [-1,1]
def resizeRearrange(frame,perc):
    ## DOWNSAMPLE ##
    scale_percent = perc # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST)
    # print('new frame size: ', resized.shape)
    ## DOWNSAMPLE ##

    # resized = frame

    ## --- ADJUST RANGE --- ##
    new_max = np.max(resized)/2
    newrange = resized/new_max -1
    # print('new range of values [' + str(np.min(newrange)) + ' ' + str(np.max(newrange)) + ']')
    ## --- ADJUST RANGE --- ##

    resized = newrange
    return resized

### ------------------------------------------------------------------------------- ###

if __name__== "__main__":
    unidim = binary_tree.CreateBinaryTree()
    index = binary_tree.SnakeOrdering()
    ord_3D, ord_triplets = binary_tree.Order3D(unidim,index)

    k=2
    n=2**k

    ### position of spatial (S), temporal (T) and spatio-temporal (ST) kernels
    ### -- and consequently projections -- in the ordered sequence
    Spos = range(1,16)
    Tpos = [31,32,63]
    STpos = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]


    folder = sorted(glob.glob('python_demo/frames/*.png'))

    perc = 100 ### --- percentage of resampling of the original frame size --- ###

    savepath = '/results/'
    print('Saving to: ', savepath)
    for f in range(len(folder)):

        ind=0
        # print('first ind of the block: ',f)
        frame = cv2.imread(folder[f], cv2.IMREAD_GRAYSCALE)
        # print(frame.shape)
        resized = resizeRearrange(frame,perc)

        block_f = np.zeros((resized.shape[0],resized.shape[1],n))
        block_f[:,:,0] = resized

        for other in range(f+1,f+n):
            ind = ind+1
            # print('other indices ', other)
            frame = cv2.imread(folder[other], cv2.IMREAD_GRAYSCALE)
            resized = resizeRearrange(frame,perc)
            block_f[:,:,ind] = resized

        count = other
        ### we obtained hte first block of n frames
        print('processing block ending at frame: ', count, end='\r')

        ### now we apply the efficient scheme, obtaining a single 2D projection for each kernel of the family
        all_filters = efficient_scheme.applyGCK(ord_3D, ord_triplets, block_f, savepath)

        rows = all_filters.shape[0]
        cols = all_filters.shape[1]

        ### Pooling over temporal and spatio-temporal projections to obtain two distinct maps
        ### see https://www.frontiersin.org/articles/10.3389/fcomp.2022.867289/full

        # avg_pooling = np.zeros((rows,cols))
        # avg_pooling = np.mean(all_filters[:,:,Tpos], axis=2)
        #
        # max_pooling = np.zeros((rows,cols))
        # max_pooling = np.max(all_filters[:,:,STpos], axis=2)

        # np.save(### --- all_filters or pooled version --- ###, savepath)
