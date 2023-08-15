import os
import pandas as pd 
### General Imports ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Autoencoder ###
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input

from tensorflow.keras.datasets import mnist
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
import numpy as np

from tensorflow.keras import callbacks
from keras.callbacks import Callback
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

############################# Data preparetion ############################# 

df = pd.read_csv('/home/kannika/code/USAI_Doctor-all_2013-2023_dummy_pathcrop_AB_imgFold.csv', dtype=str)
df = df.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1'])

# manage data sorted by column name "Folder", "category_5FPvit" ตามลำดับ
df = df.sort_values(by=['Folder', 'category_5FPvit'])
df = df[df['subset']=='train']
file_path_list_train = list(df['img_path_new'])

# Load and preprocess the input images
def load_images(file_paths, img_shape):                                                                                                                                                                                                                                                                                                                                             
    images = []
    for file_path in file_paths:
        img = load_img(file_path, color_mode='grayscale')
        img = img_to_array(img)
        img = resize(img, img_shape, mode='constant')
        img = img.astype('float32') / 255.0  # Normalize pixel values between 0 and 1
        images.append(img)
    return np.array(images)


shape_x = 224
shape_y = 224
image_shape = (shape_x, shape_y) # Change the shape based on your input images
X_train = load_images(file_path_list_train, image_shape)

from einops import rearrange, reduce, repeat

X_train_rs = rearrange(X_train, '(b1 b2) h w 1 -> b1 h w b2 ', b2=5)#np.reshape(X_test,[1250,28,28,8])
X_train_rs.shape

############################# Tensorboard ############################# 

os.chdir('/media/tohn/SSD/USAI-AE-Model')

root_logdir = '/media/tohn/SSD/USAI-AE-Model/my_logs_5FP_R1_224_500'
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir = run_logdir)


############################# Autoencoer model ############################# 
shape_x = 224
shape_y = 224

image_shape = (shape_x, shape_y) 
input_img = Input(shape=(shape_x, shape_y, 5))

# Ecoding
x = Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
x = Conv2D(16, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(16, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = UpSampling2D((4,4))(x)
x = Conv2D(8, (5, 5), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (5, 5), activation='relu', strides=(1, 1))(x)
x = UpSampling2D((4, 4))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = MaxPooling2D(pool_size=(3,3), padding='same')(x)
x = MaxPooling2D(pool_size=(3,3), padding='same')(x)
x = Conv2D(3,(3, 3), padding='same', activation='relu')(x)
encoded = MaxPooling2D(pool_size=(1, 1), padding='same')(x)

# Decoding
x = Conv2D(3,(3,3), padding='same', activation='relu', strides=(1, 1))(encoded)
x = Conv2D(3,(3,3), padding='same', activation='relu')(x)
x = Conv2D(3, (3, 3), activation='relu', strides=(1, 1))(x)
x = UpSampling2D((3, 3))(x)
x = Conv2D(3, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(3, (3, 3), activation='relu', strides=(1, 1))(x)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(3, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(3, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(3, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(3, (3, 3), activation='relu', strides=(1, 1))(x)
x = UpSampling2D((3, 3))(x)
# x = Conv2D(16, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = Conv2D(8, (3, 3), activation='relu', strides=(1, 1))(x)
x = MaxPooling2D(pool_size=(5,5), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', strides=(1, 1))(x)
x = MaxPooling2D(pool_size=(3,3), padding='same')(x)
x = Conv2D(5,(3, 3), padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Activation('linear')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.summary()

run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(run_logdir)

############################# Checkpoint ############################# 

# model checkpoint
modelName = 'AEmodel_224input_5FP_500epochs_r1.h5'
class Metrics(Callback):
    def on_epoch_end(self, epochs, logs={}):
        self.model.save(f'{modelName}')
        return

    
# For tracking Quadratic Weighted Kappa score and saving best weights
metrics = Metrics()


# Training
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mse'])


# callback
checkpoint_filepath = f'./checkpoint/'
if not os.path.exists(checkpoint_filepath) :
        os.makedirs(checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, 
                                                                   save_freq='epoch', ave_weights_only=False, monitor="mean_squared_error")

# Fit model
autoencoder.fit(X_train_rs, X_train_rs, epochs = 250, batch_size=1, validation_split = 0.1,
                callbacks = [tensorboard_cb, metrics, model_checkpoint_callback])


############################# save model ############################# 

# serialize model to JSON
model_json = autoencoder.to_json()
with open("./models/AEmodel_224input_5FP_500epochs_r1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("./models/AEmodel_224input_5FP_500epochs_r1.h5")
print("Saved model to disk")





























