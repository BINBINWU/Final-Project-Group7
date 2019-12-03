#%% Packages
from keras.models import load_model
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% User Defined Variables and cropped data
img_width = 100
img_height = 100
crop_imgs = np.load("cropped_apples.npy", allow_pickle=True)

#%% Functions
def masked_argmax(a,limit): # Defining func for regular array based soln
    valid_idx = np.where(a >= limit)[0]
    return valid_idx[a[valid_idx].argmax()]

#%% Loading the Dense Net Model
model = load_model('fruit_360_cTWO.h5')

#%% Generating and Saving Maps
y_pred=[]
count=len(crop_imgs)//30+1
for batch in range(count):

    inds_t = slice(batch * 30, (batch + 1) * 30)
    pred_pr = model.predict(crop_imgs[inds_t]/255)

    for i in pred_pr:
        try:
            y_pred.append(masked_argmax(i, 0))
        except:
            y_pred.append(121)

print(y_pred)
np.save("mapped_apples.npy", np.array(y_pred))
