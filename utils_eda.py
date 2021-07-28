import os
from pydicom import dcmread, read_file
import math
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')


def get_files(pth = None):
    '''
    iterates over path, captures a list of all the files in the path, adds them to all_files
    
    param: pth -  where to start
    return: a list of all files found including absolute path
    '''
    if pth is None:
        raise ValueError('pth cannot be null')
        
    fles=[]
    for dirname, _, filenames in os.walk(pth):
        for filename in filenames:
            fles.append(os.path.join(dirname, filename))
    return fles
    
#following from https://www.kaggle.com/kperkins411/catch-up-on-positive-samples-plot-submission-csv/edit
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
def dicom2array(path, voi_lut=True, fix_monochrome=True):
    #from https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data
    
def plot_img(img, size=(7, 7), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def plot_imgs(imgs, cols=4, size=7, is_rgb=True, title="", cmap='gray', img_size=(500,500)):
    '''
    displays images associated with dicom files in list imgs
    param: imgs - list of dicom pixel_arrays
    param img_size - if None true image size is used
    ex. 
    trn_files = uteda.get_files(PATH+'/train')
    plot_imgs([dicom2array(pth) for pth in trn_files[:15]])
    
    '''
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

import matplotlib.patches as patches
def plot_img_with_bboxes(img,  gt_bboxes=None, pred1_bboxes=None, pred2_bboxes=None, title=None,  size=7):
    '''
    displays image associated with dicom files in img
    param: img -  fullsize dicom pixel_array image
    param: gt_bboxes - ground truth
                    list of containing bounding boxe dicts, 1 per box.  Example of list with 1 box
                    ex: [{'x': 587.42021, 'y': 1377.02752, 'width': 434.11377, 'height': 196.05139}]
    param: pred1_bboxes - first set of predictions
    param: pred2_bboxes - second set of predictions
 
    '''
    h,w = img.shape
    fig,ax = plt.subplots(figsize=(size, size))
    ax.imshow(img, cmap="gray")
    
    #plot ground truth bboxes if there
    if gt_bboxes is not None:
        for i,box in enumerate(gt_bboxes):       
            # Create a Rectangle patch
            rect = patches.Rectangle((box['x'], box['y']), box['width'], box['height'], linewidth=1.5, edgecolor='w', facecolor='none', label="Ground Truth" if i ==0 else "")
            ax.add_patch(rect)
 
    #plot predicted bboxes if there
    if pred1_bboxes is not None:
        for i,box in enumerate(pred1_bboxes):       
            # Create a Rectangle patch
            rect = patches.Rectangle((box['x1'], box['y1']), box['x2']-box['x1'], box['y2']-box['y1'], linewidth=1.5, edgecolor='y', facecolor='none', label="Pred1" if i ==0 else "")
            ax.add_patch(rect)

    #plot predicted bboxes if there
    if pred2_bboxes is not None:
        for i,box in enumerate(pred2_bboxes):       
            # Create a Rectangle patch
            rect = patches.Rectangle((box['x1'], box['y1']), box['x2']-box['x1'], box['y2']-box['y1'], linewidth=1.5, edgecolor='r', facecolor='none', label="Pred2" if i ==0 else "")
            ax.add_patch(rect)

    if title is not None:
        plt.title(title)

#     ax.text(w//2,5, 'Ground truth', horizontalalignment='center', verticalalignment='top',  color='yellow', fontsize=14)
#     ax.text(w//2,50, 'Predicted', horizontalalignment='center', verticalalignment='top',  fontdict={'color':'red','size': 14 })

    ax.legend(loc='upper center')

    plt.show()

