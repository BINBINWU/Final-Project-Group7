#%% Packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import \
    Dense, \
    Dropout, \
    Flatten, \
    Conv2D, \
    MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import glob

#%% Variables
BATCH_SIZE = 32
EPOCHS = 400
LR = 1e-4
SEED = 42
np.random.seed(SEED)
IMG_WIDTH=100
IMG_HEIGHT=100
TARGET_SIZE=[IMG_WIDTH, IMG_HEIGHT]
VALID_FRUITS = [i.split("/")[-1] for i in glob.glob(
        '/home/sade/ML2_Final_Project/fruits-360/Training/*'
        )]

#%% Data Paths
train_path = r'/home/sade/ML2_Final_Project/fruits-360/Training'
test_path = r'/home/sade/ML2_Final_Project/fruits-360/Test'

#%% Data Generation
data_gen = keras.preprocessing.image.ImageDataGenerator(
    #rotation_range=90,
    width_shift_range=0.5,
    height_shift_range=0.5,
    #brightness_range=[0.5, 1],
    zoom_range=0.03,
    #channel_shift_range=0.05,
    vertical_flip=True,
    rescale=1./255
)
train_images_iter = data_gen.flow_from_directory(
    train_path,
    target_size=TARGET_SIZE,
    classes=VALID_FRUITS,
    class_mode='categorical',
    seed=SEED,
    batch_size=BATCH_SIZE
)
test_images_iter = data_gen.flow_from_directory(
    test_path,
    target_size=TARGET_SIZE,
    classes=VALID_FRUITS,
    class_mode='categorical',
    seed=SEED,
    batch_size=BATCH_SIZE
)
trained_classes_labels = list(train_images_iter.class_indices.keys())

#%% Model
model_cnn = Sequential()
model_cnn.add(Conv2D(32, kernel_size=(10, 10),
                 activation='relu',
                 input_shape=(100, 100, 3)))
model_cnn.add(Conv2D(64, (5, 5), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(4, 4)))
model_cnn.add(Dropout(0.25))
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(120, activation='softmax'))
model_cnn.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=LR),
              metrics=['accuracy'])


#%% Model training
model_cnn.fit_generator(
  # training data
  train_images_iter,

  # epochs
  steps_per_epoch = train_images_iter.n // BATCH_SIZE, #floor per batch size
  epochs = EPOCHS,

  # validation data
  validation_data = test_images_iter,
  validation_steps = test_images_iter.n // BATCH_SIZE,

  # print progress
  verbose = 1,
  callbacks = [
    #early stopping in case the loss stops decreasing
    #EarlyStopping(monitor='val_loss', patience=3),
    ModelCheckpoint(
            "/home/sade/ML2_Final_Project/fruit_360_cTWO.h5", 
            monitor='val_loss', save_best_only = True
    )
  ]
)
