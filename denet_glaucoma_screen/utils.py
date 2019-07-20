import numpy as np

from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops


def pro_process(temp_img, input_size):
    img = np.asarray(temp_img).astype('float32')
    img = np.array(Image.fromarray(img).resize((input_size, input_size)).convert(3))
    return img


def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0

    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return binary_fill_holes(np.asarray(binary).astype(int))


def Deep_Screening(target_model, tmp_img, input_size):
    temp_img = np.array(Image.fromarray(tmp_img).resize((input_size, input_size)).convert(3))
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    Pre_result = target_model.predict(temp_img)
    return Pre_result


def Disc_Crop(org_img, DiscROI_size, C_x, C_y):
    disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)
    crop_coord = [int(C_x - DiscROI_size / 2), int(C_x + DiscROI_size / 2), int(C_y - DiscROI_size / 2),
                  int(C_y + DiscROI_size / 2)]
    err_coord = [0, DiscROI_size, 0, DiscROI_size]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0]) + 1
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2]) + 1
        crop_coord[2] = 0

    if crop_coord[1] > org_img.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0]) - 1
        crop_coord[1] = org_img.shape[0]

    if crop_coord[3] > org_img.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1]) - 1
        crop_coord[3] = org_img.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[
                                                                          crop_coord[0]:crop_coord[1],
                                                                          crop_coord[2]:crop_coord[3]
                                                                          ]

    return disc_region
