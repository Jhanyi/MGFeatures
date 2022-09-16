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
from napari_segment_blobs_and_things_with_membranes import connected_component_labeling
import glob
from patchify import patchify, unpatchify
import cv2


# for SINGLE SLICE.

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
def mask_it(labels:'uint8 ndarray', organelle_mask:'ndarray'):
    '''
    Takes image label (ndarray of 2 dimensions) and binary mask of the same dimension. Returns labelled organelles corresponding to the cell.
    :param labels: ndarray
    :param organelle_mask: ndarray. non organelle should be 0 or false
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

    organelle_labels = np.zeros((labels.shape), dtype=np.uint8)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if organelle_mask[i][j] != 0:
                organelle_labels[i][j] = labels[i][j]

    # list comprehension does not work
    # organelle_labels = np.array([[labels[i][j] for i in range(labels.shape[0]) for j in range(labels.shape[1]) if organelle_mask[i][j] == True]])
    # organelle_labels = np.array([[labels[i][j] for (i, j) in zip(range(labels.shape[0]), range(labels.shape[1])) if organelle_mask[i][j] == True]])

    return organelle_labels




def img_from_tiles(folder, sl):
    '''
    A folder containing image tiles FROM 1 SEGMENTATION OBJECT exported from VAST. Has the format <filename>.vsseg_export_s%z%_Y%y%_X%x%.tif
    run this function for every segmentation object
    :param folder: folder containing 1 segmentation object exported from VAST
    :param slice: list of slice number to export. use slice='all' for exporting all slices
    :return: patched image from the tiles.
    '''

    file = r'\*_s{}_*'.format(str(sl).zfill(2))
    imgfile_list = []

    R = 0
    C = 0

    patchsize = (8192, 8192)

    for imgfile in glob.glob(folder + file):
        imgfile_list.append(imgfile)

        r = int(imgfile[-8])
        c = int(imgfile[-5])

        if r > R:
            R = r
        if c > C:
            C = c

    imageshape = (R, C, *patchsize)
    imagepatches = np.zeros(shape=imageshape, dtype=np.uint8)

    for imgfile in imgfile_list:
        r = int(imgfile[-8])
        c = int(imgfile[-5])

        imagepatches[r - 1, c - 1, :, :] = io.imread(imgfile).astype(np.uint8)

    image = unpatchify(imagepatches, (patchsize[0] * R, patchsize[1] * C))
    image = image.astype(np.uint8)

    return image

def label_cells(image):
    '''
    converts an ndarray of image to labels. Uses connected compoennt labelling. (needs to proof read with image)
    :param image: 2D image of ndarray
    :return: 2D labelled image
    '''
    binary_image = image.astype(bool)
    labels = connected_component_labeling(binary_image)
    labels = labels.astype(np.uint8)
    return labels

def ER_length(ER, labels): # put labels as global variable
    '''
    Calculates total length of ER in a cell.
    :param ER: patched ER from 1 slice. dtype bool or int
    :return: updates properties['ER_length']
    '''
    if ER.dtype != np.uint8:
        if ER.dtype == bool:
            ER = ER * 255
        ER = ER.astype(np.uint8)
    
    properties['ER_length'] = []

    # labels_draw = cv2.cvtColor(labels.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    for region in regionprops(labels):
        bbox = region.bbox  # returns minr, minc, maxr, maxc
        ER_crop = ER[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        contours, hierachy = cv2.findContours(image=ER_crop,
                                              mode=cv2.RETR_TREE,
                                              method=cv2.CHAIN_APPROX_SIMPLE
                                              )
        
        # cv2.drawContours(image=labels_draw[bbox[0]:bbox[2], bbox[1]:bbox[3]],
        #                  contours=contours,
        #                  contourIdx=-1,
        #                  color=(255, 255, 255),
        #                  thickness=1
        #                  )

        ER_len = 0
        for cnt in contours:
            ER_len_single = cv2.arcLength(cnt, True)
            # add include only if cnt[0] location isco in label number,
            ER_len += ER_len_single
            
        properties['ER_length'].append(ER_len)
    #properties['ER_length'] = np.array(properties['ER_length'])

#def count organelles

if __name__ == "__main__":
    pass
