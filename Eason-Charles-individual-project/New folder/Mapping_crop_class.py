#%% Imports
import numpy as np
import os
from PIL import Image
import math
import matplotlib.pyplot as plt

#%% Directory and Mapped
path_dir = '/home/sade/ML2_Final_Project/test multi apples'
map_pic= np.load('mapped_apples.npy', allow_pickle=True)

#%% Original image and holding list
all_files = os.listdir(path_dir)
image_files = filter(lambda x: x[-4:] == '.jpg', all_files)
image_list = list(image_files)
img_map=[]

#%% Mapping
for image in image_list:

    path = os.path.join(path_dir, image)
    img = Image.open(path)
    width, height = img.size
    kernel_sz = min(width, height)

    intal_start_left_point = 0
    intal_start_top_point = 0
    intal_start_right_point = int(kernel_sz * 0.1)
    moving_right = int(intal_start_right_point * 0.1)
    intal_start_bottom_point = int(kernel_sz * 0.1)
    moving_down = int(intal_start_bottom_point * 0.1)
    # intal_start_right_point = int(kernel_sz * 0.1)
    # moving_right = int(intal_start_right_point * 0.2)
    # intal_start_bottom_point = int(kernel_sz * 0.1)
    # moving_down = int(intal_start_bottom_point * 0.2)

    start_left_point = intal_start_left_point
    start_top_point = intal_start_top_point
    start_right_point = intal_start_right_point
    start_bottom_point = intal_start_bottom_point


    img_zero_grid = np.zeros((height,width))
    img_cor_map = {}

    i = 0
    while True:

        try:
            map_pic[i]
        except:
            break

        if map_pic[i] != 121:
        #if map_pic[i] == 49:
            if map_pic[i] not in img_cor_map.keys():
                img_cor_map[map_pic[i]] = img_zero_grid.copy()
                img_cor_map[map_pic[i]][start_top_point:start_bottom_point, start_left_point:start_right_point] += 1
            else:
                img_cor_map[map_pic[i]][start_top_point:start_bottom_point, start_left_point:start_right_point] += 1

        if start_right_point <= width:
            start_left_point += moving_right
            start_right_point += moving_right
            i += 1
        elif start_bottom_point <= height:
            start_left_point = intal_start_left_point
            start_right_point = intal_start_right_point
            start_top_point += moving_down
            start_bottom_point += moving_down
            i += 1
        else:
            break

    img_map.append(img_cor_map)

    plt.imshow(img)
    plt.show()

    for i in img_cor_map:
        path = os.path.join(path_dir, image)
        img = Image.open(path)

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(img_cor_map[i], alpha=.6)
        plt.show()

        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # bbox = img_cor_map[i].copy()
        # bbox[bbox < 10] = 0
        # bbox[bbox != 0] = 1
        # ax.imshow(bbox, alpha=.8)
        # plt.show()

        #np.mean((int(np.max(bbox)), 0))