#
import numpy as np
import scipy
import scipy.io as sio
from scipy.misc import imsave
from keras.preprocessing import image

from skimage.measure import label, regionprops
from skimage.transform import rotate 
from time import time
from utils import BW_img, Deep_Screening, Disc_Crop

import cv2

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import Model_Disc_Seg as DiscSegModel
import Model_resNet50 as ScreenModel
import Model_UNet_Side as DiscModel

Img_Seg_size = 640
Img_Scr_size = 400
ROI_Scr_size = 224

pre_model_DiscSeg = './pre_model/pre_model_DiscSeg.h5'
pre_model_img = './pre_model/pre_model_img.h5'
pre_model_ROI = './pre_model/pre_model_ROI.h5'
pre_model_flat = './pre_model/pre_model_flat.h5'
pre_model_disc = './pre_model/pre_model_disc.h5'

data_type = '.jpg'
data_img_path = './test_image/'
data_save_path = './result/'



if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

file_test_list = [file for file in os.listdir(data_img_path) if file.lower().endswith(data_type)]
print(str(len(file_test_list)))

seg_model = DiscSegModel.DeepModel(Img_Seg_size)
seg_model.load_weights(pre_model_DiscSeg, by_name=True)

img_model = ScreenModel.DeepModel(Img_Scr_size)
img_model.load_weights(pre_model_img, by_name=True)

ROI_model = ScreenModel.DeepModel(ROI_Scr_size)
ROI_model.load_weights(pre_model_ROI, by_name=True)

ROIpt_model = ScreenModel.DeepModel(ROI_Scr_size)
ROIpt_model.load_weights(pre_model_flat, by_name=True)

Disc_model = DiscModel.DeepModel(Img_Seg_size)
Disc_model.load_weights(pre_model_disc, by_name=True)

for lineIdx in range(0, len(file_test_list)):
    temp_txt = [elt.strip() for elt in file_test_list[lineIdx].split(',')]
    org_img = np.asarray(image.load_img(data_img_path + temp_txt[0]))
 
    img_scale = 2048.0 / org_img.shape[0] 
    org_img = scipy.misc.imresize(org_img, (2048, int(org_img.shape[1]*img_scale), 3))
 
    start_time = time()

    # disc segmentation
    temp_img = scipy.misc.imresize(org_img, (Img_Seg_size, Img_Seg_size, 3))
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    [prob_6, prob_7, prob_8, prob_9, prob_10] = seg_model.predict([temp_img])
    disc_map = np.reshape(prob_10, (Img_Seg_size, Img_Seg_size))

    disc_map[0:round(disc_map.shape[0] / 5), :] = 0
    disc_map[-round(disc_map.shape[0] / 5):, :] = 0
    disc_map = BW_img(disc_map, 0.5)

    regions = regionprops(label(disc_map))
    C_x = regions[0].centroid[0] * org_img.shape[0] / Img_Seg_size
    C_y = regions[0].centroid[1] * org_img.shape[1] / Img_Seg_size
    disc_region = Disc_Crop(org_img, Img_Scr_size*2, C_x, C_y)

    Disc_flat = rotate(cv2.linearPolar(disc_region, (Img_Scr_size, Img_Scr_size), Img_Scr_size, cv2.WARP_FILL_OUTLIERS), -90)

    # global screening
    Img_pred = Deep_Screening(img_model, org_img, Img_Scr_size)
    Disc_pred = Deep_Screening(ROI_model, disc_region, ROI_Scr_size)
    Polar_pred = Deep_Screening(ROIpt_model, Disc_flat, ROI_Scr_size)
    Seg_pred = Deep_Screening(Disc_model, org_img, Img_Seg_size)

    DENet_pred = np.mean([Img_pred[0][1], Disc_pred[0][1], Polar_pred[0][1], Seg_pred[0][1]])
    run_time = time() - start_time

    print('Run time: ' + str(run_time) + '   Img number: ' + str(lineIdx + 1))

    tmp_name = data_save_path+temp_txt[0]
    sio.savemat(tmp_name[:-4]+'.mat', {'Img_pred': Img_pred, 
    	'Disc_pred': Disc_pred, 'Polar_pred': Polar_pred, 'Seg_pred': Seg_pred, 'DENet_pred': DENet_pred})
    # imsave(tmp_name[:-4]+'.png', Disc_flat)

