#%% Packages
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
from joblib import Parallel, delayed

#%% Cropping funciton:
def load_training_data(path, image, IMG_SIZE):

  path = os.path.join(path, image)
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

  TEST_data = []
  TEST_label = []

  while True:
    img_crop = img.crop(box=(start_left_point, start_top_point, start_right_point, start_bottom_point))
    img_crop = img_crop.convert('RGB')
    img_crop = img_crop.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    TEST_data.append(np.array(img_crop))
    TEST_label.append(image)

    if start_right_point < width:
      start_left_point += moving_right
      start_right_point += moving_right
    elif start_bottom_point < height:
      start_left_point = intal_start_left_point
      start_right_point = intal_start_right_point
      start_top_point += moving_down
      start_bottom_point += moving_down
    else:
      break

  return np.array(TEST_data), TEST_label

#%% Parallel Cropping
def parallel_crop(path):

  IMG_SIZE = 100
  all_files = os.listdir(path)
  image_files = filter(lambda x: x[-4:] == '.jpg', all_files)
  file_list = [i for i in image_files]
  look_up = {v: i for i, v in enumerate(file_list)}

  results = Parallel(n_jobs=8)(delayed(load_training_data)(path, image, IMG_SIZE) for image in file_list)

  return(results, look_up)

# plt.imshow(results[0][0][1000], interpolation='nearest')
# plt.show()

# [base image, 0-n][images or labels, 0-1][images-> cropped image, 0-n| labels-> cropped label, 0-n]
# results[0][0][0]

if __name__ == '__main__':
  # Directory
  #path = r'/home/sade/ML2_Final_Project/fruits-360/test-multiple_fruits/'
  path = r'/home/sade/ML2_Final_Project/test multi apples/'

  # Calling Parallel Cropper
  results, look_up = parallel_crop(path)

  # Lookup index from the folder 'test multi apples'
  #interest = r'/home/sade/ML2_Final_Project/test multi apples/'
  #index = look_up[os.listdir(interest)[0]]

  # Saving apple_4 cropped images only
  np.save("cropped_apples.npy", np.array(results[0][0]))