import logging
import sys
import os
import yaml
import importlib.util

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv1D, AveragePooling1D, concatenate
from keras.utils import to_categorical
from keras import backend as K

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import numpy as np
import metrics
import visualization as vs

def classification_loss(y_true, y_pred):
    return -K.mean(K.square(y_pred), axis=-1)


#lstm_states = 32
classes = 3
timesteps = 12
pred_steps = 6
feture = 1

'''
batch_size = 256
iter_num = 2
epochs_for_temp_model1 = 10
epochs_for_temp_model2 = 8
first_epochs_for_classification_model = 4
epochs_1_for_predition_model = 16
epochs_2_for_predition_model = 8
epochs_for_classification_model = 2
last_epochs_for_predition_model = 64
'''

batch_size = 256
iter_num = 2
epochs_for_temp_model1 = 1
epochs_for_temp_model2 = 1
first_epochs_for_classification_model = 1
epochs_1_for_predition_model = 1
epochs_2_for_predition_model = 1
epochs_for_classification_model = 1
last_epochs_for_predition_model = 2

classes_list = []

def main(yaml_filepath, mode):
    """Example."""
    cfg = load_cfg(yaml_filepath)
    seed = int(cfg['train']['seed'])
    

    # Print the configuration - just to make sure that you loaded what you
    # wanted to load

    module_dataset       = load_module(cfg['dataset']['script_path'])
    module_model         = load_module(cfg['model']['script_path'])
    module_optimizer     = load_module(cfg['optimizer']['script_path'])
    module_loss_function = load_module(cfg['loss_function']['script_path'])
    module_train         = load_module(cfg['train']['script_path'])

    # scale data
    scale = float(cfg['dataset']['scale'])

    x_train, y_train, x_valid, y_valid, x_test, y_test = module_dataset.load_glucose_dataset(
        xlsx_path = cfg['dataset']['xlsx_path'],
        nb_past_steps   = timesteps,
        nb_future_steps = pred_steps,
        max_length = int(cfg['dataset']['max_length']),
        train_fraction  = float(cfg['dataset']['train_fraction']),
        valid_fraction  = float(cfg['dataset']['valid_fraction']),
        test_fraction   = float(cfg['dataset']['test_fraction']),
        sheet_pos = cfg['dataset']['sheet_pos'],
        patient_id = int(cfg['dataset']['patient_id']),
        )
    x_train *= scale
    y_train *= scale
    x_valid *= scale
    y_valid *= scale
    x_test *= scale
    y_test *= scale
    np.random.seed(seed)

    # training mode
    if mode == 'train':
        
        sequence_input = Input(shape=(timesteps, feture))
        conv1d_model = Sequential()
        conv1d_model.add(Conv1D(8, 3, padding='same', activation='relu', input_shape=(timesteps, feture)))
        conv1d_model.add(Conv1D(8, 3, activation='relu'))
        conv1d_model.add(AveragePooling1D(2))
        conv1d_model.add(Dropout(0.1))
        conv1d_model.add(Flatten())
        conv1d_model.add(Dense(classes))
        
        conv1d_feture = conv1d_model(sequence_input)
        classes_vector = Activation(activation='softmax', name='classes_vector')(conv1d_feture)
        classification_model = Model(inputs=sequence_input, output=classes_vector)
        
        lstm_states = int(cfg['model']['model_cfg']['nb_lstm_states'])
        lstm_feture_without_dropout = LSTM(lstm_states)(sequence_input)
        lstm_feture = Dropout(0.3)(lstm_feture_without_dropout)
        merged_feture = concatenate([conv1d_feture, lstm_feture])
        pred_output = Dense(1)(merged_feture)
        predition_model = Model(inputs=sequence_input, outputs=pred_output)
        
        temp_feture1 = Dense(1)(lstm_feture)
        temp_model = Model(inputs=sequence_input, outputs=temp_feture1)
        
        temp_model.compile(optimizer='adam', loss='mean_squared_error')
        classification_model.compile(optimizer='adam', loss='mean_squared_error')
        predition_model.compile(optimizer='adam', loss='mean_squared_error')
        
        #predition_model.fit(x_train, y_train,
        #            epochs=first_epochs_for_predition_model, 
        #            batch_size=batch_size,
        #            validation_data=(x_valid, y_valid))
        
        
        temp_model.fit(x_train, y_train,
               epochs=epochs_for_temp_model1, 
               batch_size=batch_size,
               validation_data=(x_valid, y_valid))

        temp_biase_weight = temp_model.layers[3].get_weights()[1]
        y_temp_train = temp_model.predict(x_train)
        y_temp_valid = temp_model.predict(x_valid)
        del temp_model
        
        temp_feture2 = Dense(1, use_bias=False, name='main_output')(conv1d_feture)
        temp_model2 = Model(inputs=sequence_input, outputs=[temp_feture2, classes_vector])
        temp_model2.compile(optimizer='adam', 
                            loss={'main_output': 'mean_squared_error', 'classes_vector': classification_loss},
                            loss_weights={'main_output': 1.0, 'classes_vector': 5})
        
        temp_model2.fit(x_train,
                        {'main_output': y_train - y_temp_train - temp_biase_weight, 'classes_vector': np.zeros((y_train.shape[0], classes))},
                        epochs=epochs_for_temp_model2, 
                        batch_size=batch_size,
                        validation_data=(x_valid, {'main_output': y_valid - y_temp_valid - temp_biase_weight, 'classes_vector': np.zeros((y_valid.shape[0], classes))}))
        del temp_model2
        del y_temp_train
        del y_temp_valid
        
        classification_model.compile(optimizer='adam', loss='mean_squared_error')
        y_clssesfication = np.argmax(classification_model.predict(x_train), axis=1)
        classes_list.append(y_clssesfication)
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
            predition_model.compile(optimizer='adam', loss='mean_squared_error')
            predition_model.fit(x_train, y_train,
                                epochs=epochs_2_for_predition_model,
                                batch_size=batch_size,
                                validation_data=(x_valid, y_valid))
            
            classification_model.compile(optimizer='adam', loss='mean_squared_error')
            y_clssesfication = np.argmax(classification_model.predict(x_train), axis=1)
            classes_list.append(y_clssesfication)
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
        
        
        
        # pos=np.argmax(classification_model.predict(x_train), axis=1)

        output_image_dir = "output_image/conv1d_plus_lstm/patient_"+str(cfg['dataset']['patient_id'])+"/"
        if os.path.exists(output_image_dir) == False:
            os.makedirs(output_image_dir)


        save_file_name_prefix = output_image_dir + "sheet_" + str(cfg['dataset']['sheet_pos'])+"_lstm_states_"+str(lstm_states)+"_loss_function_mse"+"_"
        
        #y_pred_last = model.predict(x_test)[:,-1].flatten()/scale
        y_pred = predition_model.predict(x_test)
        
        y_pred_last = y_pred[:,-1].flatten()/scale
        y_test_last = y_test[:,-1].flatten()/scale
        y_pred_t0_last = np.array([x[-1] for x in x_test])/scale
        save_file_name = save_file_name_prefix + "test.png"
        vs.plot_without_std(y_test_last, y_pred_last, 
                            title="Prediction result",
                            save_file_name = save_file_name)
            

        #y_pred_last = y_pred[:,-1].flatten()
        #y_test_last = y_test[:,-1].flatten()
        #y_pred_t0_last = np.array([x[-1] for x in x_test])
        
        rmse = metrics.root_mean_squared_error(y_test_last, y_pred_last)
        with open(os.path.join(cfg['train']['artifacts_path'], "rmse.txt"), "w") as outfile:
            outfile.write("{}\n".format(rmse))

        '''
        seg = metrics.surveillance_error(y_test_last, y_pred_last)
        with open(os.path.join(cfg['train']['artifacts_path'], "seg.txt"), "w") as outfile:
            outfile.write("{}\n".format(seg))
        '''
        
        t0_rmse = metrics.root_mean_squared_error(y_test_last, y_pred_t0_last)
        with open(os.path.join(cfg['train']['artifacts_path'], "t0_rmse.txt"), "w") as outfile:
            outfile.write("{}\n".format(t0_rmse))
        
        
        # 后面这部分观察模型在训练集和验证集上的效果
        x_test = x_train
        y_test = y_train
            
        y_pred = predition_model.predict(x_test)
        
        y_pred_last = y_pred[:,-1].flatten()/scale
        y_test_last = y_test[:,-1].flatten()/scale
        y_pred_t0_last = np.array([x[-1] for x in x_test])/scale
        save_file_name = save_file_name_prefix + "train.png"
        vs.plot_without_std(y_test_last, y_pred_last, 
                            title="Prediction result on training set", 
                            save_file_name = save_file_name)
            
        
        x_test = x_valid
        y_test = y_valid
            
        y_pred = predition_model.predict(x_test)
        
        y_pred_last = y_pred[:,-1].flatten()/scale
        y_test_last = y_test[:,-1].flatten()/scale
        y_pred_t0_last = np.array([x[-1] for x in x_test])/scale
        save_file_name = save_file_name_prefix + "validation.png"
        vs.plot_without_std(y_test_last, y_pred_last,
                            title="Prediction result on validation set",
                            save_file_name = save_file_name)
        
        print(rmse)
        

def load_module(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.exists(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


if __name__ == '__main__':
    #args = get_parser().parse_args()
    #main(args.filename, args.mode)
    filenames = 'experiments/example.yaml'
    mode = 'train'
    #mode = 'evaluate'
    main(filenames, mode)

















