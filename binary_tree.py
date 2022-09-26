#!/usr/bin/env python
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import cv2
import scipy.signal

### --- size and number of kernel --- ###
k = 2
n = 2**k
### ---size and number of kernel --- ###

# ------------------------------------------------------------------------------

### --- Binary Tree --- ###
class Node:

    def __init__(self, data, p, alfa):

        self.left = None
        self.right = None
        self.Data = data
        self.pos = p
        self.a_ind = alfa

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print( self.Data),
        if self.right:
            self.right.PrintTree()

    def createTree(self, k):
        plus = np.concatenate((self.Data, self.Data), axis=None)
        minus = np.concatenate((self.Data, self.Data*(-1)), axis=None)
        a_plus = np.concatenate((self.a_ind, [1]), axis=None)
        a_minus = np.concatenate((self.a_ind, [-1]), axis=None)
        if self.pos == 1:
            self.left = Node(plus, 1, a_plus)
            self.right = Node(minus, 2, a_minus)
        else:
            self.left = Node(minus, 1, a_minus)
            self.right = Node(plus, 2, a_plus)
        if k>1:
            k=k-1
            self.left.createTree(k)
            self.right.createTree(k)

def PrintLeaves(node, l, a):
    if(node.left is None and node.right is None):
        l.append(node.Data)
        a.append(node.a_ind)
    else:
        PrintLeaves(node.left, l, a)
        PrintLeaves(node.right, l, a)

# -------------------------------------------------------------------------------

### --- Creates the binary tree, k is the height of the tree --- ###
def CreateBinaryTree():
    root = Node(np.array([1]),1, [])
    root.createTree(k)

    unidim = []
    indices = []
    PrintLeaves(root,unidim,indices)

    return unidim
#-------------------------------------------------------------------------------
### --- Determines the snake ordering in 3D --- ###
def SnakeOrdering():
    t_dim = list(range(0, (2**k)**3))
    t_dim = np.asarray(t_dim).reshape(2**k,2**k,2**k)
    # print t_dim

    index = []

    for p in range(t_dim.shape[0]):
        i=0
        j=0
        prov = []

        prov.append(t_dim[p,i,j])
        prov.append(t_dim[p,i,j+1]) #dx
        prov.append(t_dim[p,i+1,j+1])#down
        prov.append(t_dim[p,i+1,j])#sx

        for i in range(2,n,2):
            for elem in list(t_dim[p,i,j:j+i]):
                prov.append(elem)
            for elem in list(t_dim[p,i:0:-1,j+i]):
                prov.append(elem)
            prov.append(t_dim[p,0,j+i])
            for elem in list(t_dim[p,0:i+1, j+i+1]):
                prov.append(elem)
            for elem in list(t_dim[p,i+1, j+i+1:j:-1]):
                prov.append(elem)
            prov.append(t_dim[p,i+1,0])

        if(p%2==1):
            prov.reverse()

        index.extend(np.asarray(prov))

    return index
#-------------------------------------------------------------------------------

### --- Creates tri-dim filters wrt snake ordering --- ###
def Order3D(unidim, index):
    tri_dim = []
    triplets = []
    #tri_alfa = []

    for h in range(0,len(unidim)):
        for i in range(0,len(unidim)):
            for j in range(0, len(unidim)):
                x = unidim[i]
                y = unidim[j]
                z = unidim[h]
                tri_dim.append([x[:,None,None] * y[None,:,None] * z[None,None,:]])
                triplets.append((unidim[i],unidim[j],unidim[h]))
                #tri_alfa.append(np.concatenate((indices[i],indices[j],indices[h]), axis=None))

    ord_3D = []
    #ord_aind_3D = []
    ord_triplets = []

    for l in range(0,len(index)):
        ord_3D.append(tri_dim[index[l]])
        #ord_aind_3D.append(tri_alfa[index[l]])
        ord_triplets.append(triplets[index[l]])

    return ord_3D, ord_triplets
#-------------------------------------------------------------------------------

def plot_3Dker(ker):
    def explode(data):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    # build up the numpy logo
    # n_voxels = np.zeros((4, 3, 4), dtype=bool)
    # n_voxels[0, 0, :] = True
    # n_voxels[-1, 0, :] = True
    # n_voxels[1, 0, 2] = True
    # n_voxels[2, 0, 1] = True
    n_voxels = ker
    facecolors = np.where(n_voxels==-1, '#000000', '#dedede')
    edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
    filled = np.ones(n_voxels.shape)

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    ax.view_init(elev=120, azim=0)

    plt.show()
    namefile='filter'+str(count)
    fig.savefig(namefile, bbox_inches='tight')

### --- Plots the 3D Gray-Code Kernels --- ###
def vis3D():
    count = 0
    for i in range(len(ord_3D)):
        ker = np.asarray(ord_3D[i]).reshape((n,n,n))
        print("filter "+ str(count))
        plot_3Dker(ker)
        count = count+1
