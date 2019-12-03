import numpy as np
import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from PIL import ImageFilter
import collections
from sklearn.preprocessing import LabelEncoder
import math
import random
#from keras.preprocessing.image import ImageDataGenerator

# This script generates the training set (the one you will be provided with),
# and the held out set (the one we will use to test your model towards the leaderboard).

#path_dir_test= '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/fruits-360_dataset/fruits-360/test-multiple_fruits/'
path_dir_test='/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/train-2'

#dict={'red blood cell': 0, 'ring': 1, 'schizont': 2, 'trophozoite': 3}

all_files = os.listdir(path_dir_test)
#txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
#fruit
#image_files = filter(lambda x: x[-4:] == '.jpg', all_files)
#cell
image_files = filter(lambda x: x[-4:] == '.png', all_files)




#fruit
#IMG_SIZE = 100
#cell
IMG_SIZE = 115

def load_image_data(DIR,F_index,T_index):
  TEST_data = []
  TEST_label = []
  a=list(image_files)[F_index:T_index]
  for image in a:
    #clock+=1
    #label = label_img(img)
    path = os.path.join(DIR, image)
    img = Image.open(path)
    width, height = img.size
    # plt.imshow(img)
    # plt.show()
    #img = img.convert('RGB')
    #img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    #TEST_data.append(np.array(img))

    #path = os.path.join(DIR, image)
    intal_start_left_point=0
    intal_start_top_point = 0
    # #fruit
    # intal_start_right_point = int(width*0.2)
    # moving_right = int(intal_start_right_point*0.5)
    # intal_start_bottom_point = int(height*0.2)
    # moving_down = int(intal_start_bottom_point * 0.5)
    #cell
    intal_start_right_point = int(width * 0.025)
    moving_right = int(intal_start_right_point * 0.2)
    intal_start_bottom_point = int(height * 0.025)
    moving_down = int(intal_start_bottom_point * 0.2)

    start_left_point = intal_start_left_point
    start_top_point = intal_start_top_point
    start_right_point = intal_start_right_point
    start_bottom_point = intal_start_bottom_point


    for i in range(math.ceil(width/moving_right*height/moving_down)):
      #img = Image.open(path)
      #img = img.crop(box=(0, 0, 50, 50))
      img_crop = img.crop(box=(start_left_point, start_top_point, start_right_point, start_bottom_point))
      #img = ImageOps.expand(img, (0, 0, 50, 50))
      # plt.imshow(img)
      # plt.show()
      img_crop = img_crop.convert('RGB')
      img_crop = img_crop.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
      TEST_data.append(np.array(img_crop))
      TEST_label.append(image)

      if start_right_point <= width:
        start_left_point+=moving_right
        start_right_point+=moving_right
      elif start_bottom_point <= height:
        start_left_point = intal_start_left_point
        start_right_point = intal_start_right_point
        start_top_point += moving_down
        start_bottom_point += moving_down
      else:
        break
    # if clock==count:
    #   break
  #shuffle(train_data)
  return np.array(TEST_data)#,TEST_label

#load image, due to storage limit need input index to select
data=load_image_data(path_dir_test,713,723)


np.save("x_train_cell.npy", np.array(data)) #; np.save("y_train.npy",label)
# np.save("x_test.npy", np.array(x_test)); np.save("y_test.npy", np.array(y_test))


