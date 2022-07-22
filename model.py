import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dropout, Dense, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import math

# define all the hyper-parameter
batch_size = 128
learning_rate = 0.0003
epochs = 7
drop_rate = 0.5

# read in the images path and steering labels from csv
data_path, label = preprocessing.read_csv('driving_log.csv')

# split the training and validation data
train_data, validation_data, train_label, validation_label = train_test_split(data_path, label, test_size=0.2)

# create train and validation generator
train_generator = preprocessing.generator(train_data, train_label, batch_size=batch_size)
validation_generator = preprocessing.generator(validation_data, validation_label, batch_size=batch_size)

# create the model
model = Sequential()

# crop and norm
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((40, 20), (0, 0))))

# conv layers
# 100 320 3
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='elu'))
model.add(BatchNormalization())
# 48 158 24
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='elu'))
model.add(BatchNormalization())
# 22 77 36
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='elu'))
model.add(BatchNormalization())
# 9 37 48
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 2), activation='elu'))
model.add(BatchNormalization())
# 7 18 64
model.add(Conv2D(filters=80, kernel_size=3, strides=(1, 1), activation='elu'))
model.add(BatchNormalization())
# 5 16 80
model.add(Conv2D(filters=96, kernel_size=3, strides=(1, 1), activation='elu'))
model.add(BatchNormalization())
# 3 14 96
model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='elu'))
model.add(BatchNormalization())
# 1 12 128
# flatten layer
model.add(Flatten())
# fully connect layers
# in 1526 | out 100
model.add(Dense(100, activation='elu'))
model.add(Dropout(rate=drop_rate))
# in 100 | out 50
model.add(Dense(50, activation='elu'))
model.add(Dropout(rate=drop_rate))
# in 50 | out 10
model.add(Dense(10, activation='elu'))
model.add(Dropout(rate=drop_rate))
# in 10 | out 1
model.add(Dense(1))

# create the optimizer
opt = Adam(lr=learning_rate)

# compile the model
model.compile(loss='mse', optimizer=opt)

# callback function to save the model
checkpoint = ModelCheckpoint(filepath='./model.h5', monitor='val_loss', save_best_only=True)
# run the model with generator
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_data) / batch_size),
                    validation_data=validation_generator,
                    validation_steps=math.ceil(len(validation_data) / batch_size),
                    epochs=epochs, verbose=1, callbacks=[checkpoint])
