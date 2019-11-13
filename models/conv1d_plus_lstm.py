from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv1D, AveragePooling1D, concatenate
#from keras.utils import to_categorical
#from keras import backend as K

def load_with_lstm_states(input_shape, output_shape, states):
    sequence_input = Input(shape=input_shape)
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
    
    lstm_feture_without_dropout = LSTM(states)(sequence_input)
    lstm_feture = Dropout(0.3)(lstm_feture_without_dropout)
    merged_feture = concatenate([conv1d_feture, lstm_feture])
    pred_output = Dense(1)(merged_feture)
    predition_model = Model(inputs=sequence_input, outputs=pred_output)
    return conv1d_model, classification_model, predition_model