import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from skimage.measure import regionprops
from scipy import sparse
import matplotlib.patches as mpatches
import numpy.ma as ma
import napari
from numba import njit

#ignore for now
def get_bbox(labels:'uint8 ndarray'):
    fig, ax = plt.subplots(figsize=(10, 6))

    assert type(labels) == np.ndarray
    ax.imshow(labels)

    for region in regionprops(labels):
        #TODO: add index
        minr, minc, maxr, maxc = region.bbox

        #TODO: remove or change for validation
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)

        ax.add_patch(rect)
        ax.plot(minc, minr, 'or')
        ax.plot(minc, maxr, 'or')
        ax.plot(maxc, maxr, 'or')
        ax.plot(maxc, minr, 'or')

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

@njit
def mask_it(labels:'uint8 ndarray', organelle_mask:'binary ndarray'):
    # assert type(labels) == np.ndarray # why dosent work with numba?
    # assert type(organelle_mask) == np.ndarray and (organelle_mask.dtype == bool or np.bool)

    # nroi = labels.max()
    # for i in range(nroi):

    # organelle_mask = ~organelle_mask
    # organelle_mask = organelle_mask * 1
    #
    # organelle_labels = ma.array(labels, mask = organelle_mask)
    # io.imshow(organelle_labels)

    organelle_labels = np.zeros((labels.shape))

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if organelle_mask[i][j] == True:
                organelle_labels[i][j] = labels[i][j]


    # list comprehension does not work
    # organelle_labels = np.array([[labels[i][j] for i in range(labels.shape[0]) for j in range(labels.shape[1]) if organelle_mask[i][j] == True]])
    # organelle_labels = np.array([[labels[i][j] for (i, j) in zip(range(labels.shape[0]), range(labels.shape[1])) if organelle_mask[i][j] == True]])


    #timeit
    return organelle_labels

def



if __name__ == "__main__":
    import time
    labels = io.imread(r'label.tiff')
    # get_bbox(labels)
    organelle_mask = io.imread('lys.tiff')

    start = time.perf_counter()
    organelle_labels = mask_it(labels, organelle_mask)
    end = time.perf_counter()

    print('time: {}s'.format(end-start))

    io.imsave('organelle_labels_3.tiff', organelle_labels)
