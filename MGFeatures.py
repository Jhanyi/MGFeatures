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
from skimage.measure import regionprops

properties = {}

def draw_bbox(bbox):
    '''
    Takes a tuple of x and y values for bounding box. Returns an array of coordinates to be passed as data points in napari shape layer
    :param bbox: Tuple of (minr, minc, maxr, maxc). returned from skimage's region.bbox.
    :return: 4 x 2 array of path length of a rectangle
    '''
    rect = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
    return rect

# for plotting in io - ignore this
# def get_bbox(labels:'uint8 ndarray'):
    # fig, ax = plt.subplots(figsize=(10, 6))
    #
    # assert type(labels) == np.ndarray
    # ax.imshow(labels)
    #
    # for region in regionprops(labels):
    #     minr, minc, maxr, maxc = region.bbox
    #
    #     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                               fill=False, edgecolor='red', linewidth=2)
    #
    #     ax.add_patch(rect)
    #     ax.plot(minc, minr, 'or')
    #     ax.plot(minc, maxr, 'or')
    #     ax.plot(maxc, maxr, 'or')
    #     ax.plot(maxc, minr, 'or')
    #
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()


@njit
def mask_it(labels:'uint8 ndarray', organelle_mask:'binary ndarray'):
    '''
    Takes image label (ndarray of 2 dimensions) and binary mask of the same dimension. Returns labelled organelles corresponding to the cell.
    :param labels: ndarray
    :param organelle_mask: ndarray bool
    :return: organelles labelled with their corresponding cells
    '''
    # assert type(labels) == np.ndarray # why assert dosen't work with numba?
    # assert type(organelle_mask) == np.ndarray and (organelle_mask.dtype == bool or np.bool)

    #trying masked array - does not work
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

    return organelle_labels

def img_from_tiles(folder, slice='all'):
    '''
    A folder containing image tiles FROM 1 SEGMENTATION OBJECT exported from VAST. Has the format <filename>.vsseg_export_s%z%_Y%y%_X%x%.tif
    run this function for every segmentation object
    :param folder: folder containing 1 segmentation object exported from VAST
    :param slice: list of slice number to export. use slice='all' for exporting all slices
    :return: patched image from the tiles.
    '''
    pass

def ER_length(ER):
    '''
    Calculates total length of ER in a cell.
    :param ER: patched ER from 1 slice. dtype bool or int
    :return: updates properties['ER_length']
    '''
    if ER.dtype != np.uint8:
        if ER.dtype == bool:
            ER = ER * 255
        ER = ER.astype(np.uint8)

    ER


#def count organelles

if __name__ == "__main__":
    pass
