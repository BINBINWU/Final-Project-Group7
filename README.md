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
## Final-Group-Presentation - PowerPoint slideshow detailing our work.
## Final-Group-Project-Report - Latex report of our project.
## Group-Proposal - Out initial proposal.
## Code - All the code used in our project.
