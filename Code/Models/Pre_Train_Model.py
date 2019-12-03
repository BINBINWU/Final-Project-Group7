from keras.applications import DenseNet121
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,Dropout
from keras.layers import GlobalAveragePooling2D
import numpy as np



img_width = 100
img_height = 100

#path_dir_train = '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/fruits-360_dataset/fruits-360/Training/'
#path_dir_validate = '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/fruits-360_dataset/fruits-360/Test/'
#path_dir_test= '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/fruits-360_dataset/fruits-360/test-multiple_fruits/'

generator = ImageDataGenerator(rescale=1./255)

train_generator = generator.flow_from_directory(
    path_dir_train,
    target_size=(img_width, img_height),
    batch_size=128,
    interpolation="lanczos",
    class_mode='categorical')


validation_generator = generator.flow_from_directory(
    path_dir_validate,
    target_size=(img_width, img_height),
    batch_size=128,
    interpolation="lanczos",
    class_mode='categorical')

test_generator = generator.flow_from_directory(
    path_dir_test,
    target_size=(img_width, img_height),
    batch_size=128,
    interpolation="lanczos",
    class_mode='categorical'
)

base_model=DenseNet121(weights='imagenet', include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(2048,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.2)(x)
preds=Dense(120,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=True
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    # samples_per_epoch=num_train,
    nb_epoch=30,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint("/home/ubuntu/Deep-Learning/Keras_/pre_pc.h5", monitor="val_loss", save_best_only=True)]
    # nb_val_samples=num_validate
)