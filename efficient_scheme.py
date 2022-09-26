#!/usr/bin/env python
import binary_tree

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import cv2
import scipy.signal
import time
import glob

### Function that look for the first different element between two alpha-related kernels
def findFirst(u,v):
    if len(u)==len(v):
        for index in range(len(u)):
            if u[index]!=v[index]:
                return index

### Implementation of the efficient scheme
def applyGCK(ord_3D, ord_triplets, block, savepath):

    all_filters = np.zeros((block.shape[0],block.shape[1],len(ord_3D)))

    ### Given a family of n GCKs, we will use a spatial convolution on our image with the first Kernels
    ### And then apply all the other kernels with the efficient scheme
    n = block.shape[2]
    B01 = np.zeros_like((block.shape[0], block.shape[1], block.shape[2]))
    B01 = scipy.signal.fftconvolve(block, np.asarray(ord_3D[0]).reshape((n,n,n)), mode='full')
    block_size = B01.shape[2]

    ## normalization step ##
    central = B01[1:B01.shape[0]-2,1:B01.shape[1]-2,(block_size//2)]
    norm_B01 = (central - np.min(central)) / (np.max(central)-np.min(central))
    ##

    all_filters[:,:,0] = norm_B01

    for ker in range(1, len(ord_3D)):

        v0,v1,v2 = ord_triplets[ker-1]
        v3,v4,v5 = ord_triplets[ker]

        ### find DELTA, direction and ordering of v_p and v_m
        if np.array_equal(np.outer(v0,v2),np.outer(v3,v5)):
            direction = 0
            DELTA = findFirst(v1,v4)
            # DELTA = diff
            if(v1[DELTA] == -1):
                vp_vm = 0
            else:
                vp_vm = 1

        elif np.array_equal(np.outer(v1,v2), np.outer(v4,v5)):
            direction = 1
            DELTA = findFirst(v0,v3)
            # DELTA = diff[0][0]
            if(v0[DELTA] == -1):
                vp_vm = 0
            else:
                vp_vm = 1

        elif np.array_equal(np.outer(v0,v1),np.outer(v3,v4)):
            direction = 2
            DELTA = findFirst(v2,v5)
            # DELTA = diff[0][0]
            if(v2[DELTA] == -1):
                vp_vm = 0
            else:
                vp_vm = 1

        B_02 = np.zeros_like(B01)
        dim1,dim2,dim3 = B01.shape

        if direction == 0:
            for i in range(0,dim2):
                if(i < DELTA):
                    B_02[:,i,:] = B01[:,i,:]
                else:
                    if vp_vm == 0:
                        B_02[:,i,:] = B01[:,i,:] + B01[:,i-DELTA,:] + B_02[:,i-DELTA,:]
                    else:
                        B_02[:,i,:] = B01[:,i,:] - B01[:,i-DELTA,:] - B_02[:,i-DELTA,:]
        elif direction == 1:
            for i in range(0,dim1):
                if(i < DELTA):
                    B_02[i,:,:] = B01[i,:,:]
                else:
                    if vp_vm == 0:
                        B_02[i,:,:] = B01[i,:,:] + B01[i-DELTA,:,:] + B_02[i-DELTA,:,:]
                    else:
                        B_02[i,:,:] = B01[i,:,:] - B01[i-DELTA,:,:] - B_02[i-DELTA,:,:]
        else:
            for i in range(0,dim3):
                if(i < DELTA):
                    B_02[:,:,i] = B01[:,:,i]
                else:
                    if vp_vm == 0:
                        B_02[:,:,i] = B01[:,:,i] + B01[:,:,i-DELTA] + B_02[:,:,i-DELTA]
                    else:
                        B_02[:,:,i] = B01[:,:,i] - B01[:,:,i-DELTA] - B_02[:,:,i-DELTA]
        B01 = B_02
        ## normalization step ##
        central = B_02[1:B_02.shape[0]-2,1:B_02.shape[1]-2,(block_size//2)]
        norm_B02 = (central - np.min(central)) / (np.max(central)-np.min(central))
        ##

        all_filters[:,:,ker] = norm_B02

    # returns all the 2D projections, one for each kernel of the family
    return all_filters

# ### Function that resize the original video frames by a certain percentage and adjust the range of values of the images to be in [-1,1]
# def resizeRearrange(frame,perc):
#     ## DOWNSAMPLE ##
#     scale_percent = perc # percent of original size
#     width = int(frame.shape[1] * scale_percent / 100)
#     height = int(frame.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     resized = cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST)
#     # print('new frame size: ', resized.shape)
#     ## DOWNSAMPLE ##
#
#     # resized = frame
#
#     ## --- ADJUST RANGE --- ##
#     new_max = np.max(resized)/2
#     newrange = resized/new_max -1
#     # print('new range of values [' + str(np.min(newrange)) + ' ' + str(np.max(newrange)) + ']')
#     ## --- ADJUST RANGE --- ##
#
#     resized = newrange
#     return resized
#
# ### ------------------------------------------------------------------------------- ###
#
# if __name__== "__main__":
#     unidim = binary_tree.CreateBinaryTree()
#     index = binary_tree.SnakeOrdering()
#     ord_3D, ord_triplets = binary_tree.Order3D(unidim,index)
#
#     ### position of spatial (S), temporal (T) and spatio-temporal (ST) kernels
#     ### -- and consequently projections -- in the ordered sequence
#     Spos = range(1,16)
#     Tpos = [31,32,63]
#     STpos = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]
#
#
#     folder = ### --- path of the folder containing the video frames --- ###
#
#     perc = 100 ### --- percentage of resampling of the original frame size --- ###
#
#     savepath = ### --- path of the output folder --- ###
#     print('Saving to: ', savepath)
#     for f in range(len(folder)):
#
#         ind=0
#         # print('first ind of the block: ',f)
#         frame = cv2.imread(folder[f], cv2.IMREAD_GRAYSCALE)
#         # print(frame.shape)
#         resized = resizeRearrange(frame,perc)
#
#         block_f = np.zeros((resized.shape[0],resized.shape[1],n))
#         block_f[:,:,0] = resized
#
#         for other in range(f+1,f+n):
#             ind = ind+1
#             # print('other indices ', other)
#             frame = cv2.imread(folder[other], cv2.IMREAD_GRAYSCALE)
#             resized = resizeRearrange(frame,perc)
#             block_f[:,:,ind] = resized
#
#         count = other
#         ### we obtained hte first block of n frames
#         print('processing block ending at frame: ', count, end='\r')
#
#         ### now we apply the efficient scheme, obtaining a single 2D projection for each kernel of the family
#         all_filters = applyGCK(ord_3D, ord_triplets, block_f, savepath)
#
#         rows = all_filters.shape[0]
#         cols = all_filters.shape[1]
#
#         ### Pooling over temporal and spatio-temporal projections to obtain two distinct maps
#         ### see https://www.frontiersin.org/articles/10.3389/fcomp.2022.867289/full
#
#         avg_pooling = np.zeros((rows,cols))
#         avg_pooling = np.mean(all_filters[:,:,Tpos], axis=2)
#
#         max_pooling = np.zeros((rows,cols))
#         max_pooling = np.max(all_filters[:,:,STpos], axis=2)
#
#         np.save(### --- all_filters or pooled version --- ###, savepath)
