import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.applications import MobileNet, VGG16
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import math

#model = load_model('/home/ubuntu/Deep-Learning/Keras_/pre_pc.h5')
model = load_model('/home/ubuntu/Deep-Learning/Keras_/MLP/Exam_1/mlp_bwu.hdf5')

#path_dir_train = '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/fruits-360_dataset/fruits-360/Training/'
path_dir_train = '/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/train-2'

#print(model.summary())
def masked_argmax(a,limit): # Defining func for regular array based soln
    valid_idx = np.where(a >= limit)[0]
    return valid_idx[a[valid_idx].argmax()]

# #fruit
# img_width = 100
# img_height = 100
#cell
img_width = 115
img_height = 115

# generator = ImageDataGenerator(rescale=1./255)

# test_generator = generator.flow_from_directory(
#     path_dir_train,
#     target_size=(img_width, img_height),
#     batch_size=128,
#     interpolation="lanczos",
#     class_mode='categorical',
#     shuffle=False
# )

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
x= np.load("/home/ubuntu/Deep-Learning/Keras_/CNN/Object_Dect/x_train_cell.npy",allow_pickle=True)
# history = model.fit_generator(
#     train_generator,
#     # samples_per_epoch=num_train,
#     nb_epoch=30,
#     validation_data=validation_generator,
#     callbacks=[ModelCheckpoint("/home/ubuntu/Deep-Learning/plant-classification/pre_pc_more.h5", monitor="val_loss", save_best_only=True)]
#     # nb_val_samples=num_validate
# )

y_pred=[]
count=len(x)//30+1
for batch in range(count):
    inds_t = slice(batch * 30, (batch + 1) * 30)
    # #fruit
    # Y_pred = model.predict(x[inds_t]/255)
    #cell
    Y_pred = model.predict((x[inds_t] / 255).reshape(len(np.array(x[inds_t])), -1))

    for i in Y_pred:
        try:
            y_pred.append(masked_argmax(i, 0.8))
        except:
            y_pred.append(121)

print(y_pred)
np.save("x_map_cell.npy", np.array(y_pred))
#print('Classification Report')
#target_names = ['Asclepias tuberosa', 'Cercis canadensis', 'Cichorium intybus', 'Cirsium vulgare', 'Claytonia virginica', 'Gaillardia pulchella', 'Glechoma hederacea', 'Liquidambar styraciflua', 'Lonicera japonica', 'Lotus corniculatus', 'Parthenocissus quinquefolia', 'Phytolacca americana', 'Prunella vulgaris', 'Prunus serotina', 'Rosa multiflora', 'Rudbeckia hirta', 'Taraxacum officinale', 'Trifolium pratense', 'Verbascum thapsus', 'Viola sororia']
#print(classification_report(test_generator.classes, y_pred, target_names=target_names))

