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
#     print(f'original_images_detected_on_dir= {original_images_detected_on_dir}')
#     print(f'marked_up_images_dest_dir {marked_up_images_dest_dir}')
#     print(f'Run names={run_names}')
    for i in tqdm(range(max_rows_to_eval)):
        row=df.loc[i]

        img_name = row.loc['id']+img_suffix

        img=load_img( original_images_detected_on_dir, img_name)
        img=np.array(img)
        height,width = img.shape
#         print(f'Height={height} Width={width}')     

        #ground truth bounding boxes
        gt_boxes=get_boxes(row)
#         print(gt_boxes)

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
   
#     print(f'In get_pred_bboxes, run_names = {run_names}')
    for dir in run_names:
        #convert bounding boxes into lists of floats
        pred_boxes_and_confidence=[]
        
        #file to open
        fle = run_dir+dir+ '/labels/' + img_name.split('.')[0] +'.txt'
#         print(f'   file to open = {fle}')
        if not os.path.isfile(fle):
            print(f'Missing label file for image {img_name} for run {dir}' )
            results.append([])
            continue
            
        with open(fle) as f:
            lines=f.readlines()
#             print(f'        file contains={lines}')
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
    pth=os.path.join(pth_destdir , imagename)
    return Image.open(pth)
    
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
    OFFSETX=0
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
 

    cmap=['y','r','g','b', 'c', 'm']  #[ 'm', , 'k']
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
        


import shutil
import os
from os.path import join, getsize
import json
from ensemble_boxes import *


class BBManager():
    '''
    point at detect dir for yolo and run it,
    it will return wbf for a particular image
    sample run
    #
    pth = '/home/keith/runs/detect'
    a =BBManager(pth, runs={'exp0':1,'exp1':2,'exp2':3})
    info1=a._get_bboxes('1111')
    info2=a._get_bboxes('EEEEE')    #fails empty dict
    wbf_boxes1 = a.get_wbf_boxes('1111') #calculates and returns list of all wbf boxes for this image
    wbf_boxes2 = a.get_wbf_boxes('1111') #returns prev calculated value
    wbf_boxes3 = a.get_wbf_boxes('1111',tmp_run_weights=[3,4], force_recalc=True) #recalculates for this image
    wbf_boxes4 = a.get_wbf_boxes('EEEE') #fails  empty list
    a.do_all_imgs()
    '''
    def __init__(self,pth,runs,dir_to_find='labels', ext_to_find="txt"  ):
        '''

        :param pth: where to start looking
        :param runs: dict of runs to look for in path and the relative importence of each run
                     ex. runs={'expo':.9,'exp1':.7}
        :param dir_to_find:
        :param ext_to_find:
        '''
        self.pth = pth
        self.runs=runs
        self.dir_to_find = dir_to_find
        self.ext_to_find = ext_to_find
        # self.cur_pth = pth
        self.init()

    def init(self):
        self.all_files = []
        self.images_set=set()
        self.all_images = {}

        self._get_all_files(self.pth)
        self._parse_files()
        
    def _get_all_files(self,pth, dir=''):
        '''

        :param pth: start here
        :param dir_to_find: looking for this dir
        :param ext_to_find: looking for files of this type in above dir
        :return:
        '''
        pth=join(pth,dir)
        for root, dirs, files in os.walk(pth):
            if dir == self.dir_to_find:
                fls = [join(pth,fle) for fle in files if fle.split('.')[-1] == self.ext_to_find]
                self.all_files.extend(fls)
                return
            for dir in dirs:
                self._get_all_files(pth,dir)


    def _get_boxes_from_fle(self,fn):
        with open(fn) as f:
            bboxes=[]
            lines=f.readlines()
            for lne in lines:
                lne=lne.replace('\n','')
                lne=lne.replace('  ',' ')
                lne="[" +lne.replace(' ',' , ') +"]"
                lne=json.loads(lne)
                bboxes.append(lne)
            return bboxes


    def _parse_files(self):
        for fle in self.all_files:
            data = fle.split('/')
            img = data[-1].split('.')[0]
            exp = data[-3]
            if exp not in self.runs:
                continue
            self.images_set.add(img)
            bbox = {}

            boxes = self._get_boxes_from_fle(fle)

            if self.all_images.get(img) is None:
                self.all_images[img] = {}
            self.all_images[img][exp] = boxes

    def _get_bboxes(self, img):
        if self.all_images.get(img) is None:
            return {}
        return self.all_images[img]

    @staticmethod
    def _convert_from_center_width_height_to_x1y1x2y2(bbox):
        '''
        The submisison requires xmin, ymin, xmax, ymax format.
        YOLOv5 returns x_center, y_center, width, height
        :param bboxes:
        :return: boxes
        '''
        xc,yc,w,h=bbox

        xmin = xc - (w / 2)
        ymin = yc - (h / 2)
        xmax = xc + (w / 2)
        ymax = yc + (w / 2)
        lst=[xmin, ymin, xmax, ymax]
        if len([v for v in lst if v<0])>0 or len([v for v in lst if v>1])>0:
            raise ValueError("OH NO!wh->x1y1x2y2, values <0 or >1! "+str(lst))
 
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def __convert_from_x1y1x2y2_to_center_width_height(bbox):
        x1,y1,x2,y2=bbox
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w / 2
        yc = y1 + h / 2
        lst=[xc, yc, w, h]
        if len([v for v in lst if v<0])>0:
            print("OH NO!,x1y1x2y2->wh values negative! "+str(lst))
        return lst


    @staticmethod
    def __fix( x):
        '''
        fixes a zipped list where x[1] is also a list
        :param x:
        :return:
        usage ex.
        c=[5,6,7]
        b=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]
        s=[.1,.2,.3]
        tmp=list(map(fix,zip(c,b,s)))
        #print(tmp)
        #[[5, 1, 1, 1, 1, 0.1], [6, 2, 2, 2, 2, 0.2], [7, 3, 3, 3, 3, 0.3]]
        '''
        x[1].insert(0, x[0])
        x[1].append(x[2])
        return x[1]

    def get_wbf_boxes(self, img, out_put_dir, iou_thr=0.55, skip_box_thr=0.00, run_nms=True, run_wbf=True):
        ''' 
         applies WBF and maybe NMS to all the runs for an image and returns the adjusted bboxes

        :param img: string, image to find boxes for

         :param iou_thr: IoU value for boxes to be a match
        :param skip_box_thr: exclude boxes with score lower than this variable
        :param force_recalc force recalc, (use if change params)
        :return: list, wbf bboxes in Yolov5 format (class, xc,yc,w,h, conf)
        '''
        #returns a dict
        bboxes=self._get_bboxes(img)
           
        #get the runs for this i mage
        runs=[*bboxes]
        
        #get possible run weights
        run_weights=[self.runs[val] for val in runs]
#         print(run_weights)
       
        #what if no boxes there?
        if len(bboxes)==0:
            return []

        #get rid of wbf if it's there since we are recalculating it
        if out_put_dir in bboxes:
            del bboxes[out_put_dir]

        boxes_list = []
        conf_list = []
        labels_list = []

        for key, val in bboxes.items():
            v_conf=[]
            v_bboxes=[]
            v_labels=[]
 
            for lst in val:
                # change coordinates
                try:                   
                    l=self._convert_from_center_width_height_to_x1y1x2y2(lst[1:5])
                except ValueError as err:
                    print(err.args)
                    continue
                v_conf.append(lst[5])
                v_labels.append(lst[0])                  
                v_bboxes.append(l)

            boxes_list.append(v_bboxes)
            conf_list.append(v_conf)
            labels_list.append(v_labels)

#         print(boxes_list)
#         print(conf_list)
#         print(labels_list)
#         print(f'for img {img} Before wbf,Length boxes={len(boxes_list)}, scores={len(conf_list)}, labels={len(labels_list)}, weights={len(run_weights)}')
        if run_wbf==True:
            boxes_list, conf_list, labels_list = weighted_boxes_fusion(boxes_list, conf_list, labels_list, weights=run_weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#         print(f'After wbf,Length boxes={len(boxes)}, scores={len(scores)}, labels={len(labels)}\n')

        if  len(boxes_list)>0 and run_nms == True:
            if run_wbf==True:
                boxes_list=[boxes_list]
                conf_list = [conf_list]
                labels_list= [labels_list]
                
            nb_wbf=len(boxes_list)
#             print(f'{len(boxes_list)}')
#             for box in boxes_list:               
#                 print(box)
                
            #NOTE: for this competition there are no overlapping boxes AFAICT, so if there are any overlapping boxes, drop the low 
            #       conf ones
            #NOTE I explicitly do NOT pass run weights since wbf above will reduce the nSumber of runs to 1, the out_put_dir one
            boxes_list, conf_list, labels_list = nms(boxes_list, conf_list, labels_list, iou_thr=.4)
            if(len(boxes_list) != nb_wbf):
#                 print(f'NMS dropped {nb_wbf- len(boxes)} boxes in file {img}')
                pass
        else:
            print(f'Image {img} has no boxes remaining above {skip_box_thr}!')
     
        # reassemble
        #convert back to yolo format
        n_boxes=[]
        for box in boxes_list:
            n_boxes.append(self.__convert_from_x1y1x2y2_to_center_width_height(box))

        #zip it back together
        boxes = list(map(self.__fix, zip(labels_list, n_boxes, conf_list)))
        bboxes[out_put_dir]=boxes

        return bboxes[out_put_dir]

    def do_all_imgs(self,drop_below_this_conf=None,skip_box_thr=0.001,  run_nms=True, run_wbf=True, out_put_dir='wbf', verbose=False ):
        '''
        :param run_weights: list, how to weigh each runs bboxes, ie if you are looking at runs=['run1', 'run2'] and
               run_weights = [2,1] then run1 is twice as  important as run2,

        :param drop_below_this_conf: drop all boxes below this confidence level after running get_wbf_boxes 
        run_nms if true run nms after wbf (gets rid of most overlapping bounding boxes)

        creates a wbf/labels dir in self.path,
        then saves all bounding boxes for all images in .txt file with img name
        :return: nothing
        '''
        self.init()
     
        #create a place for the output
        wbf_dir = os.path.join(self.pth,out_put_dir)
        try:
            shutil.rmtree(wbf_dir)
        except os.error:
            pass

        wbf_dir=os.path.join(wbf_dir,'labels')
        os.makedirs(wbf_dir, exist_ok=True)

        dropped_bacause_low_conf=0
        for img in self.images_set:
            wbf_boxes = self.get_wbf_boxes(img,out_put_dir=out_put_dir, skip_box_thr=skip_box_thr,run_nms=run_nms, run_wbf=run_wbf )            
            
            #lets get rid of any boxes within any other boxes
            data=''
            numb_dropped_boxes=0
            for box in wbf_boxes:
                #dump low confidence boxes if we want
                if drop_below_this_conf is not None:
                    if box[5]<drop_below_this_conf:
                        numb_dropped_boxes+=1
                        if verbose:
                            print(f'{img} dropped bbox, conf {box[5]}<drop_below_this_conf of {drop_below_this_conf}')
                        continue

                box = [round(num, 6) for num in box]
                box[0]=int(box[0])

                box=str(box)
                box = box.replace('[','')
                box = box.replace(']', '')
                box=box.replace(',','')            
                data=data+box+ '\n'
             
            if data == '':
                if verbose:
                    print(f'Image {img} started with {len(wbf_boxes)} bboxes, all dropped because confidence < {drop_below_this_conf} no .txt file generated')
                pass
            else:
                if verbose:
                    print(f'Image {img} started with {len(wbf_boxes)} bboxes, dropped {numb_dropped_boxes} bboxs (confidence < {drop_below_this_conf})')
                with open(os.path.join(wbf_dir,(img+'.txt')), "w") as f:
                    f.write(data) 
############################################################## mAP and AP calculations

def convert(bbox,row):
    '''
    bbox is a dict
    take the correctly sized bbox in yolo format (x_center, y_center, width, height)
    convert to xmin,xmax,ymin,ymax
    then scale values between 0 and 1
    return: list correct bbox
    '''
    #scale factor
    orig_width=row['dim1']
    orig_height=row['dim0']
    
    #convert
    xmin= ((bbox['x'])/1)/orig_width
    xmax= ((bbox['x'] +bbox['width']/2)/1)/orig_width
    ymin= ((bbox['y'] )/1)/orig_height
    ymax= ((bbox['y'] + bbox['height']/2)/1)/orig_height
   
    return[xmin,xmax,ymin,ymax]
                            

def get_gt_df_for_mAP_calc(df_src,df_dst, col):
    '''
    damn metadata
    dim0 is y
    dim1 is x
    '''
 
    for _, row in df_src.iterrows(): 
        br=pd.Series(data=[np.nan for _ in range(6)], index=df_dst.columns)
        br['id']=row['id']

        #how many boxes
        boxes=get_boxes(row)

        #if no boxes enter nan for this row plus the id!
        if len(boxes) == 0:
            df_dst=df_dst.append(br, ignore_index=True)
        else:
            #otherwise create 1 row per bounding box
            br['class']=0
            for box in boxes:
                #normalize
                br['xmin'],br['xmax'],br['ymin'],br['ymax'] = convert(box,row)
                df_dst=df_dst.append(br, ignore_index=True)               
    return df_dst


def get_predict_df_for_mAP_calc(df_dst, path_yolo_detect='tmp/yolov5/runs/detect', pred_dir='wbf_only'):
    #use this class to get info
    run_info={pred_dir: 1.0}
    wbf=BBManager(path_yolo_detect, runs=run_info)
    wbf.all_images
    for key, value in wbf.all_images.items():
        br=pd.Series(data=[np.nan for _ in range(7)], index=df_dst.columns)
        br['id'] =key
#         print(f'{key}')
        for _, val1 in value.items():
            for val2 in val1:
                #finally 1 row per pred in yolov5 format  - class, xc,yc,w,h, conf)
                br['class']=0.0
                br['conf']=val2[5]
                br['xmin'],br['ymin'],br['xmax'],br['ymax'] = wbf._convert_from_center_width_height_to_x1y1x2y2(val2[1:5])
                df_dst=df_dst.append(br, ignore_index=True) 
    return df_dst
        
#SAMPLE USAGE
#see test_map.ipynb
#!pip install map-boxes 

# df_valid = pd.DataFrame(columns=['id', 'class','xmin','xmax','ymin','ymax'])
# df_valid = uteda.get_gt_df_for_mAP_calc(df_src,df_valid, 'boxes')

# df_predict_cols = pd.DataFrame(columns=['id', 'class','conf','xmin','xmax','ymin','ymax'])
# df_predict = uteda.get_predict_df_for_mAP_calc(df_predict, path_yolo_detect='tmp/yolov5/runs/detect', pred_dir='wbf_only')

# from map_boxes import mean_average_precision_for_boxes

# ann = valid_df[['id', 'class','xmin','xmax','ymin','ymax']].values
# det = df_predict[['id', 'class','conf','xmin','xmax','ymin','ymax']].values
# mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det,iou_threshold=0.5,verbose=True)
# print(f'mAP={mean_ap}, average precision={average_precisions}')