import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as impimg

from keras.utils.np_utils import to_categorical #one hot encoding 
from sklearn.model_selection import train_test_split
np.random.seed(2)
from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator



#loading data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train = train.drop(labels = ["label"],axis = 1)
y_train = train["label"]

#normalization
x_train = x_train/255.0
test = test/255.0

#reshape
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#label encoding - one hot
y_train = to_categorical(y_train , num_classes = 10)

#split train-test data
random_seed = 2
x_train , x_val , y_train , y_val = train_test_split(x_train,y_train,test_size = 0.1,random_state = random_seed)

#cnn layer setting
model = Sequential()
model.add(Conv2D(filters = 32 , kernel_size = (5,5) , padding = 'Same', activation= 'relu' , input_shape = (28,28,1)))
model.add(Conv2D(filters = 32 , kernel_size = (5,5) , padding = 'Same', activation= 'relu' , input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64 , kernel_size = (3,3) , padding = 'Same', activation= 'relu' , input_shape = (28,28,1)))
model.add(Conv2D(filters = 64 , kernel_size = (3,3) , padding = 'Same', activation= 'relu' , input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))

#optimizer
optimizer = RMSprop()
#complie the model
model.compile(optimizer = optimizer , loss = ["categorical_crossentropy"],metrics = ["accuracy"])

#annealer - learning rate 
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 3 , verbose = 2 , factor = 0.5,min_lr = 0.00001 )
#epochs
epochs = 100
batch_size = 128


#data agumentation - to incearse the data and to overcome overfitting 
datagen = ImageDataGenerator(rotation_range = 10,zoom_range = 0.1, width_shift_range = 0.1,height_shift_range = 0.1)
datagen.fit(x_train)

#fit the model
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = epochs,validation_data = (x_val,y_val),verbose = 2 ,steps_per_epoch = x_train.shape[0]//batch_size, callbacks = [learning_rate_reduction])

#evaluation of the model - training and validation curves- val_acc > training_acc - no overfitting
fig ,ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'],color = 'b',label = 'training_loss')
ax[0].plot(history.history['val_loss'],color = 'r',label = 'validation_loss',axes = ax[0])
legend = ax[0].lengend(loc = 'best',shadow = True)

ax[1].plot(history.history['acc'],color = 'b',label = 'training_acc')
ax[1].plot(history.history['val_acc'],color = 'r',label = 'validation_acc')
legend = ax[1].legend(loc = 'best',shadow = True)

#predict result 
results = model.predict(test)
results = np.argmax(results,axis =1)
results = pd.Series(results, name= 'Label')

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis =1)
submission.to_csv("cnn_mnist_datagen.csv",index = False)




















