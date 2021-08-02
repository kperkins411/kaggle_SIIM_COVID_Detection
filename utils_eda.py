import os
from pydicom import dcmread, read_file
import math
import matplotlib.pyplot as plt
import cv2
import warnings
import random
import numpy as np
import torch
import pandas as pd
from PIL import Image
import os
import json
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

class Config:
    n_folds: int = 5
    seed: int = 42
    num_classes: int = 2 
    img_size: int = 256
    fold_num: int = 0
    device: str = 'cuda:0'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("Running seed_everything... DONE!")
    
seed_everything(Config.seed)


def get_all_dataframes(til_csv = 'input/siim-covid19-detection/train_image_level.csv', 
                  tsl_csv='input/siim-covid19-detection/train_study_level.csv',
                  tss_csv='input/siim-covid19-detection/sample_submission.csv',
                  meta_csv = 'input/siim-covid19-resized-to-512px-png/meta.csv'):
    '''
    USE THIS WHEN USING PRETRAINED RESIZED IMAGES FROM XHULU!
    get all the training images info, strip id and study field of suffix's,
    get metadata and merge them
    returns: df with original image dimensions in it'''

    #load data
    image_df = pd.read_csv(os.path.join(os.getcwd(),til_csv), index_col=None)
    study_df=pd.read_csv(os.path.join(os.getcwd(),tsl_csv), index_col=None)
    submission = pd.read_csv(os.path.join(os.getcwd(), tss_csv), index_col=None)

    # clean data
    image_df['id']=image_df['id'].map(lambda x: x.split('_')[0])   
    study_df['id']=study_df['id'].map(lambda x: x.split('_')[0])
    
    # Load meta.csv file
    # Original dimensions are required to scale the bounding box coordinates appropriately.
    meta_df = pd.read_csv(os.path.join(os.getcwd(),meta_csv))

    cols=meta_df.columns
    cols=[col for col in cols]
    cols[0]='id'
    meta_df.columns=cols

    # Merge both the dataframes
    image_df= image_df.merge(meta_df, on='id',how="left")
    
    return image_df, study_df, submission
    
# The submisison requires xmin, ymin, xmax, ymax format. 
# YOLOv5 returns x_center, y_center, width, height
def correct_bbox_format(bboxes, orig_width, orig_height):
    correct_bboxes = []
    for b in bboxes:
        xc, yc = int(np.round(b[1]*orig_width)), int(np.round(b[2]*orig_height))
        w, h = int(np.round(b[3]*orig_width)), int(np.round(b[4]*orig_height))

        xmin= xc - int(np.round(w/2))
        ymin= yc - int(np.round(h/2))
        xmax= xc + int(np.round(w/2))
        ymax= yc + int(np.round(h/2))
        conf= b[5]
        
        correct_bboxes.append([xmin, ymin, xmax, ymax, conf])
        
    return correct_bboxes

# Scale the bounding boxes according to the size of the resized image. 
def scale_bbox(row, bboxes, img_x, img_y):
    # Get scaling factor
    scale_x = img_x/row.dim1
    scale_y = img_y/row.dim0
    
    for bbox in bboxes:
        bbox['x'] = int(np.round(bbox['x']*scale_x, 4))
        bbox['y'] = int(np.round(bbox['y']*scale_y, 4))
        bbox['width'] = int(np.round(bbox['width']*(scale_x), 4))
        bbox['height']= int(np.round(bbox['height']*scale_y, 4))
        
    return bboxes

def generate_images(df,max_rows_to_eval, original_images_detected_on_dir,marked_up_images_dest_dir, run_names, show_image, img_suffix='.png', yolov5_preds_dir ='./tmp/yolov5/runs/detect/' ):
    '''
    catch all to generate and save images that are marked up with ground truth bounding boxes and prediction bounding boxes
    df - if using XULU resized images then usea get_all_dataframes_dir and pass in path to meta.csv with orig image size
         this will record orig image dimensions in df so bounding boxes can be correctly sized
    max_rows_to_eval
    original_images_detected_on_dir- where images that were detected on are (and have bounding box label files)
    marked_up_images_dest_dir - where marked up images are placed
    run_names - list of run(s) that generated bboxes that you are interested in, any or all found in yolov5/runs/detect
    show_image
    img_suffix='.png'
    '''
#     print(original_images_detected_on_dir)
#     print(marked_up_images_dest_dir)
#     print(run_names)
    for i in tqdm(range(max_rows_to_eval)):
        row=df.loc[i]

        img_name = row.loc['id']+img_suffix

        img=load_img( original_images_detected_on_dir, img_name)
        img=np.array(img)
        height,width = img.shape
        print(f'Height={height} Width={width}')     

        #ground truth bounding boxes
        gt_boxes=get_boxes(row)
        print(gt_boxes)

        #lets see if df1 has better h,w info
        if 'dim0' in row:
            gt_boxes = scale_bbox(row, gt_boxes, width, height)
              
        #get predicted bounding boxes
        results = get_pred_bboxes(img_name,original_images_detected_on_dir, run_names, run_dir =yolov5_preds_dir,   orig_width=width, orig_height=height)
#         print(results)
        
        results1=dict(zip(run_names,results)) 

        #plot it with the b_boxes
        plot_img_with_bboxes(img,img_name, gt_boxes,results1, size=15, out_path=marked_up_images_dest_dir,show_image=show_image)

        
def get_pred_bboxes(img_name,img_dir, run_names, run_dir,orig_width=None, orig_height=None):
    '''
    img_name name + suffix  (ex. a.jpg)
    img_dir dir the image is in
    run_names names of the runs of interest in a list[exp15, exp16]
    run_dir where the detect runs oututs are stored (like yolov5/runs/detect/)
    get all the bounding boxes for img that are stored in multiple run directories
    '''
    if(orig_width is None or orig_height is None):
        im= load_img( img_dir, img_name)
        im=np.array(im)
        orig_height,orig_width = im.shape
    
    results=[]
    for dir in run_names:
        #convert bounding boxes into lists of floats
        pred_boxes_and_confidence=[]
        
        #file to open
        fle = run_dir+dir+ '/labels/' + img_name.split('.')[0] +'.txt'
        if not os.path.isfile(fle):
            print(f'Missing label file for image {img_name} for run {dir}' )
            results.append([])
            continue
            
        with open(fle) as f:
            lines=f.readlines()
            for lne in lines:
                lne=lne.replace('\n','')
                lne="[" +lne.replace(' ',',') +"]"
                lne=json.loads(lne)               
                pred_boxes_and_confidence.append(lne)
        pred_boxes = correct_bbox_format(pred_boxes_and_confidence,orig_width,orig_height)
        
        #convert to a dict
        keys=["x1","y1","x2","y2","conf"]
        for i,b in enumerate(pred_boxes):
            pred_boxes[i]=dict(zip(keys,b))
        results.append(pred_boxes)
    return results

def copy_dicom_img_to_dir(row, pth_dicom_fles, pth_destdir):
    '''
    row - pandas series
    pth_dicom_fles - 'input/siim-covid19-detection/train/' for ex
    pth_destdir - where all images will wind up (like ''./test_tmp/'')
    return im.shape(height,width) to be logged
    
    ex.
    # copy imag to test dir
    for i in range(MAX_ROWS):
         copy_dicom_img_to_dir(df1.loc[i],TRAIN_DIR ,TEST_DIR )
    
    '''       
    study=  row.loc['StudyInstanceUID']
    dcm_file=row.loc['id']
    
    #create a path to the study
    pth =  pth_dicom_fles + row.loc['StudyInstanceUID']

    #get all dicom files from the study
    dcms = get_files(pth) 

    if (len(dcms)>1):
        #find the correct image
        dcms=list(filter(lambda x:dcm_file in x, dcms))
   
    #get the image
    img=dicom2array(dcms[0])
    
    #save it to path
    if not os.path.exists(pth_destdir):
        os.mkdir(pth_destdir)

    im = Image.fromarray(img)     
    im.save(pth_destdir+dcm_file+'.png')
    return img.shape

def load_img( pth_destdir, imagename):
    '''
    just loads an image
    ex.
    nme = df1.loc[0,'id']+'.png'
    im= load_img( TEST_DIR, nme)
    '''
    return Image.open(pth_destdir + imagename)
    
def get_boxes(row):
    '''
    Convert the string that contaings bounding boxes 
    into a list of dicts and return
    ex.
    # get boxes
    all_boxes=[]
    for i in range(MAX_ROWS):
        all_boxes.append(get_boxes(df1.loc[i]))
    '''

    if (pd.isnull(row.loc['boxes'])):
        return []
    
    boxes=row.loc['boxes'].replace('\'','"')
    return json.loads(boxes)
def populate_number_bounding_boxes_column(df):
    '''
    go through dataset, and add a field indicating number of bounding boxes present for that row
    look at boxes field, count {'s to get the number of bboxes. add to field n_bboxes

    returns dataset with new populated field 
    '''
    if 'n_bboxes' not in df.columns:
        #get list of number bboxes per row
        f=lambda x:x.count('{') if type(x) is str else 0
        numb_bboxes_per_row=list(map(f,df.boxes))

        #add bboxes column
        df['n_bboxes']=numb_bboxes_per_row    
    return df

def display_bb_distribution(df):
    '''
    go through dataset, and add a field indicating number of bounding boxes for that row
    0- means no opacity
    >0 opacity
    prints out some percentage info
    Use the field 
    returns dataset with field 
    '''
    #get list of number bboxes per row
    f=lambda x:x.count('{') if type(x) is str else 0
    numb_bboxes_per_row=list(map(f,df.boxes))    
 
    m=max(numb_bboxes_per_row)
    print(f'Maximum number of bounding boxes={m}\n')
    
    tots=0
    percentages=[]
    
    for i in range(0, m+1):
        tot = numb_bboxes_per_row.count(i)
        print(f'number images with {i} bounding boxes={tot}')
        percentages.append(tot)
        tots+=tot
    print(f'\nTotal images={tots}, total with bounding boxes={tots-numb_bboxes_per_row.count(0)}\n')

    percentages=[x/tots for x in percentages]
    for i,pct in enumerate(percentages):
        print(f'{"{:.2f}".format(pct)} % has {i} bounding boxes')
    return 

def get_files(pth = None):
    '''
    iterates over path, captures a list of all the files in the path, adds them to all_files
    
    param: pth -  where to start
    return: a list of all files found including absolute path
    '''
    if pth is None:
        raise ValueError('pth cannot be Snull')
        
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
def plot_img_with_bboxes(img,img_name,  gt_bboxes=None, pred_bboxes=None,   size=7, out_path="output/", show_image=True):
    '''
    displays image associated with dicom files in img
    param: img -  fullsize dicom pixel_array image
           img_name - name of the image
    param: gt_bboxes - ground truth
                    list of containing bounding boxe dicts, 1 per box.  Example of list with 1 box
                    ex: [{'x': 587.42021, 'y': 1377.02752, 'width': 434.11377, 'height': 196.05139}]
    param: pred_bboxes - predictions of form
    param: pred2_bboxes - second set of predictions
 
    '''
    OFFSETX=100
    offsetx=0
    
    OFFSETY=0
    h,w = img.shape
    fig,ax = plt.subplots(figsize=(size, size))
    ax.imshow(img, cmap="gray")
    
    #plot ground truth bboxes if there
    if gt_bboxes is not None:
        for i,box in enumerate(gt_bboxes):       
            # Create a Rectangle patch
            rect = patches.Rectangle((box['x'], box['y']), box['width'], box['height'], linewidth=1.5, edgecolor='w', facecolor='none', label="Ground Truth" if i ==0 else "")
            ax.add_patch(rect)
 
    cmap=['y','r','g','b', 'c']
    c=0
    #plot predicted bboxes if there
    if pred_bboxes is not None:
        for key,val in pred_bboxes.items():
            for i,box in enumerate(val):       
                # Create a Rectangle patch
                rect = patches.Rectangle((box['x1'], box['y1']), box['x2']-box['x1'], box['y2']-box['y1'], linewidth=1.5, edgecolor=cmap[c], facecolor='none', label=key if i ==0 else "")
                ax.add_patch(rect)
                ax.text(box['x1']+offsetx,box['y1'], box['conf'], bbox=dict(fill=False, edgecolor=cmap[c], linewidth=2))
            c+=1
            offsetx+=OFFSETX

    plt.title(img_name)

    ax.legend(loc='upper center')
    
    #save for later display
    #save it to path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    fig.savefig((out_path+img_name),bbox_inches='tight')

    if(show_image == True):
        plt.show()
    else:
        plt.close(fig)
        

