import numpy as np
import os
from PIL import Image
import math
import matplotlib.pyplot as plt

##frist
#path_dir_test= '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/fruits-360_dataset/fruits-360/test-multiple_fruits/'
#cell
path_dir_test = '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/train-2'

map_pic= np.load("/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/x_map_cell.npy",allow_pickle=True)

all_files = os.listdir(path_dir_test)
image_files = filter(lambda x: x[-4:] == '.png', all_files)

def map_image_data(DIR,F_index,T_index):
    a=list(image_files)[F_index:T_index]
    img_map=[]
    endpoint=0
    for image in a:
        # label = label_img(img)
        path = os.path.join(DIR, image)
        img = Image.open(path)
        width, height = img.size
        plt.imshow(img)
        plt.show()
        # img = img.convert('RGB')
        # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        # TEST_data.append(np.array(img))

        # path = os.path.join(DIR, image)
        intal_start_left_point = 0
        intal_start_top_point = 0
        # intal_start_right_point = int(width * 0.1)
        # moving_right = int(intal_start_right_point * 0.4)
        # intal_start_bottom_point = int(height * 0.1)
        # moving_down = int(intal_start_bottom_point * 0.4)
        # cell
        intal_start_right_point = int(width * 0.025)
        moving_right = int(intal_start_right_point * 0.2)
        intal_start_bottom_point = int(height * 0.025)
        moving_down = int(intal_start_bottom_point * 0.2)

        start_left_point = intal_start_left_point
        start_top_point = intal_start_top_point
        start_right_point = intal_start_right_point
        start_bottom_point = intal_start_bottom_point
        img_zero_grid=np.zeros((height,width))
        img_cor_map={}
        for i in range(endpoint,endpoint+math.ceil(width / moving_right * height / moving_down)):
            # img = Image.open(path)
            # img = img.crop(box=(0, 0, 50, 50))
            # img = img.crop(box=(start_left_point, start_top_point, start_right_point, start_bottom_point))
            if map_pic[i]!=121:
                if map_pic[i] not in img_cor_map.keys():
                    img_cor_map[map_pic[i]]=img_zero_grid.copy()
                    img_cor_map[map_pic[i]][start_top_point:start_bottom_point, start_left_point:start_right_point] += 1
                else:
                    img_cor_map[map_pic[i]][start_top_point:start_bottom_point,start_left_point:start_right_point]+=1

            if start_right_point <= width:
                start_left_point += moving_right
                start_right_point += moving_right
            elif start_bottom_point <= height:
                start_left_point = intal_start_left_point
                start_right_point = intal_start_right_point
                start_top_point += moving_down
                start_bottom_point += moving_down
            else:
                endpoint=i
                break

        img_map.append(img_cor_map)
    return img_map

cor_map=map_image_data(path_dir_test,713,723)

for i in cor_map:
    for j in i.keys():
        plt.imshow(i[j])
        plt.show()

