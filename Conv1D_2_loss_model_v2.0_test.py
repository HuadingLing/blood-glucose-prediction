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
    
    
lstm_states = 64
classes = 8
timesteps = 12
feture = 1

batch_size = 128
iter_num = 0
epochs_for_temp_model1 = 32
epochs_for_temp_model2 = 32
first_epochs_for_classification_model = 16
epochs_1_for_predition_model = 16
epochs_2_for_predition_model = 16
epochs_for_classification_model = 16
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

temp_feture1 = Dense(1)(lstm_feture)
temp_model = Model(inputs=sequence_input, outputs=temp_feture1)

#pms(conv1d_model)
#pms(classification_model)
#pms(temp_model)
#pms(predition_model)

temp_model.compile(optimizer='adam', loss='mean_squared_error')
classification_model.compile(optimizer='adam', loss='mean_squared_error')
predition_model.compile(optimizer='adam', loss='mean_squared_error')


x_train = np.random.randn(100, timesteps, feture)
y_train = np.random.randn(100, feture)
y_clssesfication = np.random.randn(x_train.shape[0], classes)
yy=[]

x_valid = np.random.randn(10, timesteps, feture)
y_valid = np.random.randn(10, feture)

x_test = np.random.randn(10, timesteps, feture)
y_test = np.random.randn(10, feture)

temp_model.fit(x_train, y_train,
               epochs=epochs_for_temp_model1, 
               batch_size=batch_size,
               validation_data=(x_valid, y_valid))

temp_biase_weight = temp_model.layers[3].get_weights()[1]
y_temp = temp_model.predict(x_train)
#del temp_model

temp_feture2 = Dense(1, use_bias=False)(conv1d_feture)
temp_model2 = Model(inputs=sequence_input, outputs=temp_feture2)
temp_model2.compile(optimizer='adam', loss='mean_squared_error')

temp_model2.fit(x_train,
                y_train - y_temp - temp_biase_weight,
                epochs=epochs_for_temp_model2, 
                batch_size=batch_size,
                validation_data=(x_valid, y_valid))
#del temp_model2
del y_temp

y_clssesfication = np.argmax(classification_model.predict(x_train), axis=1)
yy.append(y_clssesfication)
classification_model.fit(x_train, 
                         to_categorical(y_clssesfication, num_classes=classes), 
                         epochs=first_epochs_for_classification_model,
                         batch_size=batch_size)

for i in range(iter_num):
    conv1d_model.trainable = False
    predition_model.layers[1].trainable = True
    predition_model.compile(optimizer='adam', loss='mean_squared_error')
    predition_model.fit(x_train, y_train,
                        epochs=epochs_1_for_predition_model,
                        batch_size=batch_size,
                        validation_data=(x_valid, y_valid))
    
    conv1d_model.trainable = True
    predition_model.layers[1].trainable = False
    # predition_model.layers[5].trainable = False ???
    predition_model.compile(optimizer='adam', loss='mean_squared_error')
    predition_model.fit(x_train, y_train,
                        epochs=epochs_2_for_predition_model,
                        batch_size=batch_size,
                        validation_data=(x_valid, y_valid))
    
    y_clssesfication = np.argmax(classification_model.predict(x_train), axis=1)
    yy.append(y_clssesfication)
    classification_model.compile(optimizer='adam', loss='mean_squared_error')
    classification_model.fit(x_train, 
                         to_categorical(y_clssesfication, num_classes=classes), 
                         epochs=epochs_for_classification_model,
                         batch_size=batch_size)
    
conv1d_model.trainable = False
predition_model.layers[1].trainable = True
predition_model.compile(optimizer='adam', loss='mean_squared_error')
predition_model.fit(x_train, y_train,
                    epochs=last_epochs_for_predition_model,
                    batch_size=batch_size,
                    validation_data=(x_valid, y_valid))


'''
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
'''




