from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv1D, AveragePooling1D, concatenate, Lambda, Add
from keras.utils import to_categorical
from keras import backend as K
import numpy as np

def pms(model):
    print()
    print(model.summary())
    print()
    print(model.inputs)
    print(model.outputs)
    print()
    for c in model.get_weights():
        print(c.shape)
    print()
    
'''  
def mean_squared_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(y_pred - y_true), axis=-1)
'''
def classification_loss(y_true, y_pred):
    return -K.mean(K.square(y_pred), axis=-1)
    
    
lstm_states = 128
classes = 8
timesteps = 12
feture = 1

iter_num = 0
batch_size = 64
first_epochs_for_predition_model = 16
epochs_for_classification_model = 16
epochs_for_predition_model = 16
last_epochs_for_classification_model = 32
last_epochs_for_predition_model = 64

np.random.seed(0)

sequence_input = Input(shape=(timesteps, feture))
conv1d_model = Sequential()
conv1d_model.add(Conv1D(8, 3, padding='same', activation='relu', input_shape=(timesteps, feture)))
conv1d_model.add(Conv1D(8, 3, activation='relu'))
conv1d_model.add(AveragePooling1D(2))
conv1d_model.add(Dropout(0.1))
conv1d_model.add(Flatten())
conv1d_model.add(Dense(classes))

conv1d_feture = conv1d_model(sequence_input)
classes_vector = Activation(activation='softmax')(conv1d_feture)
classification_model = Model(inputs=sequence_input, output=classes_vector)

lstm_feture_without_dropout = LSTM(lstm_states)(sequence_input)
lstm_feture = Dropout(0.3)(lstm_feture_without_dropout)
merged_feture = concatenate([conv1d_feture, lstm_feture])
pred_output = Dense(1)(merged_feture)
predition_model = Model(inputs=sequence_input, outputs=pred_output)

#pms(conv1d_model)
#pms(classification_model)
#pms(predition_model)

classification_model.compile(optimizer='adam', loss='mean_squared_error')
predition_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


x_train = np.random.randn(1000, timesteps, feture)
y_train = np.random.randn(1000, feture)
y_clssesfication = np.random.randn(x_train.shape[0], classes)
yy=[]

x_valid = np.random.randn(100, timesteps, feture)
y_valid = np.random.randn(100, feture)

x_test = np.random.randn(100, timesteps, feture)
y_test = np.random.randn(100, feture)

predition_model.fit(x_train, y_train,
                    epochs=first_epochs_for_predition_model, 
                    batch_size=batch_size,
                    validation_data=(x_valid, y_valid))

for i in range(iter_num):
    y_clssesfication = np.argmax(classification_model.predict(x_train), axis=1)
    yy.append(y_clssesfication)
    classification_model.fit(x_train, to_categorical(y_clssesfication, num_classes=classes), 
                             epochs=epochs_for_classification_model,
                             batch_size=batch_size)
    predition_model.fit(x_train, y_train,
                        epochs=epochs_for_predition_model,
                        batch_size=batch_size,
                        validation_data=(x_valid, y_valid))

y_clssesfication = np.argmax(classification_model.predict(x_train), axis=1)
yy.append(y_clssesfication)
classification_model.fit(x_train, to_categorical(y_clssesfication, num_classes=classes), 
                             epochs=last_epochs_for_classification_model,
                             batch_size=batch_size)

conv1d_model.trainable = False
# Or: predition_model.layers[2].trainable = False
# Or: classification_model.layers[1].trainable = False

#classification_model.compile(optimizer='adam', loss=classification_loss)
predition_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
predition_model.fit(x_train, y_train, 
                    epochs=last_epochs_for_predition_model,
                    batch_size=batch_size,
                    validation_data=(x_valid, y_valid))

# pos=np.argmax(classification_model.predict(x_train), axis=1)








