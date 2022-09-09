from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import Sequential
from keras.layers import  Conv1D, Dense,Activation ,Dropout, Flatten,MaxPooling2D,Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D,RepeatVector
import cv2, pathlib, splitfolders
from tensorflow.keras.models import load_model

# create model
from tensorflow.python.keras.saving.saved_model.load import recurrent

model = Sequential()

model.add(Conv2D(64,(3, 3) , activation='relu', input_shape=(150,150, 3), padding='same', strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(RepeatVector(13))
model.add(recurrent.LSTM(128, return_sequences=True, input_shape=(150,150, 3), activation='relu'))
model.add(recurrent.LSTM(64))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()

# Compile model
adam = Adam(lr=0.00001)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


epochs = 30
batch_size = 64
img_height, img_width = 150, 150
input_shape = (img_height, img_width, 3)


def create_data(data_base):
    data_base = pathlib.Path(data_base)
    splitfolders.ratio(data_base, output='images/', seed=1234, ratio=(0.8, 0.1, 0.1), group_prefix=None)
    data_generator = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = data_generator.flow_from_directory('images/train/', target_size=(img_height, img_width),
                                                         class_mode='categorical', batch_size=batch_size,
                                                         subset='training')
    valid_generator = data_generator.flow_from_directory('images/val/', target_size=(img_height, img_width),
                                                         class_mode='categorical', batch_size=batch_size, shuffle=False)
    test_generator = data_generator.flow_from_directory('images/test/', target_size=(img_height, img_width),
                                                        class_mode='categorical', batch_size=batch_size, shuffle=False)
    return train_generator, valid_generator, test_generator


train_data, valid_data, test_data = create_data('C:/Users/amanz/PycharmProjects/stage/data/data_augm_clt1')
import flwr as fl
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()
    def fit(self,parameters,config):
        model.set_weights(parameters)
        history_1 = model.fit_generator(generator=train_data, epochs=1,validation_data=valid_data)
        print("Fit history : ",history_1.history)
        return model.get_weights(),len(train_data),{}
    def evaluate(self,parameters,config):
        model.set_weights(parameters)
        print("hey hhhh",model.evaluate_generator(test_data, steps=16))
        train_loss, train_acc = model.evaluate_generator(generator=test_data, steps=16)
        print('Eval accuracy : ',train_acc)
        return train_loss,len(test_data),{"accuracy":train_acc}

fl.client.start_numpy_client(server_address="localhost:8080",
                             client=FlowerClient(),
                             grpc_max_message_length=1024*1024*1024)

