# Final-Project-Group7

The purpose of our project was to determine if there was a quick and simple way
to generate bounding boxes for object detection using an already trained artificial
neural network. Specifcally, we wanted a way to do this without the need of human
annotation or the need of backpropagation to calculate bounding boxes. Our solution
was to use a sliding window that is fed into an already trained ANN to generate
binary maps. These binary maps are then mapped onto the original image and
summed to generate a full map that reports object location and shape, we call this
the Cropping-Predicting-Mapping (CPM) algorithm. Our solution is unique in its
simplicity, efficiency, and the ability to use any already constructed ANN for object
boundary identification purposes. Furthermore, there are a few limitations and many
avenues for further research to make this approach robust.

# Folders:
Final-Group-Presentation - PowerPoint slideshow detailing our work.

Final-Group-Project-Report - Latex report of our project.

Group-Proposal - Out initial proposal.

Code - All the code used in our project.

# Data:
The Fruit-360 data used can be obtained from: https://www.kaggle.com/moltean/fruits

or here:https://storage.cloud.google.com/fruits-360/fruits.zip?authuser=1

Some additional images used are in this repo under the "Code" section.

The Cells data used can be obtained from: https://data.broadinstitute.org/bbbc/BBBC041/

The individual cells training data can be found here: https://storage.googleapis.com/exam-deep-learning/train.zip

# Code:
The code is executed in this order:

1. Train our model (or use your own) using: Predicting_crop_class.py (for Fruit-360 data)

2. Obtain the cropped images from the image of interest using: Parallel_Cropping.py or Cropping.py

3. Obtain the predicted classes from the cropped images using: Predicting_crop_class.py or Predicting.py

4. Map the predictions onto the original images of interest using: Mapping_crop_class.py or Mapping.py
