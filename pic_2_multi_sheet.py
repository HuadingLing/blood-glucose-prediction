import logging
import sys
import os
import yaml
import importlib.util

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM
from datasets.ohio import load_glucose_dataset

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import numpy as np
import metrics
import visualization as vs

xlsx_path = "data/unprocessed_cgm_data.xlsx"
future_steps = 6
feature = 1
scale = 0.0025
seed = 0

true_id=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,27,29,30,31,32,33,35,37,38,39,40,41,43,44,45,47,48,50]

patient_num = 41
sheet_num = 6

batch_size = 256
epochs = 200

n_past = [6,12,18,24] #长度<=7
LSTM_states = [8,16,32,64,128]

#True_id = [5,9,11,12]

def main(yaml_filepath, mode):
    rmse_list = []
    for pid in [13, 18]: # 5,9,10,11,12,14,18,19,20,24,29,30,32,33,35
        patient_true_id = true_id[pid-1]
        rmse_result = []
        output_image_dir = "output_image/pic_2_multi_sheet/patient_"+str(patient_true_id)+"/"
        if os.path.exists(output_image_dir) == False:
            os.makedirs(output_image_dir)
        
        for past_steps in n_past:
            for sheet in range(sheet_num - 1):
                x_train_temp, y_train_temp, x_valid, y_valid, x_test, y_test = load_glucose_dataset(
                        xlsx_path = xlsx_path,
                        nb_past_steps   = past_steps,
                        nb_future_steps = future_steps,
                        max_length = 256,
                        train_fraction  = 1.0,
                        valid_fraction  = 0.0,
                        test_fraction   = 0.0,
                        sheet_pos = sheet,
                        patient_id = pid,
                        )
                if sheet == 0:
                    x_train = x_train_temp
                    y_train = y_train_temp
                else:    
                    x_train = np.concatenate((x_train, x_train_temp), axis=0)
                    y_train = np.concatenate((y_train, y_train_temp), axis=0)
                
            
            x_train_temp, y_train_temp, x_valid, y_valid, x_test, y_test = load_glucose_dataset(
                    xlsx_path = xlsx_path,
                    nb_past_steps   = past_steps,
                    nb_future_steps = future_steps,
                    max_length = 256,
                    train_fraction  = 0.8,
                    valid_fraction  = 0.0,
                    test_fraction   = 0.2,
                    sheet_pos = sheet_num - 1,
                    patient_id = pid,
                    )
            
            x_train = np.concatenate((x_train, x_train_temp), axis=0)
            y_train = np.concatenate((y_train, y_train_temp), axis=0)
            
            x_train *= scale
            y_train *= scale
            x_test *= scale
            y_test *= scale
            
            for lstm_states in LSTM_states:
                np.random.seed(seed)
        
                sequence_input = Input(shape=(past_steps, feature))
                
                lstm_feture_without_dropout = LSTM(lstm_states)(sequence_input)
                lstm_feture = Dropout(0.1)(lstm_feture_without_dropout)
                pred_output = Dense(1)(lstm_feture)
                predition_model = Model(inputs=sequence_input, outputs=pred_output)
                
                predition_model.compile(optimizer='adam', loss='mean_squared_error')
                
                predition_model.fit(x_train, y_train, 
                                    epochs=epochs,
                                    batch_size=batch_size)

                #save_file_name_prefix_1 = output_image_dir + "past_steps_" + str(past_steps) + "_lstm_states_" +str(lstm_states)+ "_"
                
                y_pred = predition_model.predict(x_test)
                del predition_model
                
                y_pred_last = y_pred[:,-1].flatten()/scale
                y_test_last = y_test[:,-1].flatten()/scale
#                save_file_name = save_file_name_prefix_1 + "test.png"
#                vs.plot_without_std(y_test_last, y_pred_last, 
#                                    title="Prediction result",
#                                    save_file_name = save_file_name)
                
                rmse = metrics.root_mean_squared_error(y_test_last, y_pred_last)
                rmse_result.append(rmse)
        
        vs.plot_mse(rmse_result, LSTM_states, n_past, 'Patient ' + str(patient_true_id), output_image_dir + 'result.pdf')
        rmse_list.append(rmse_result)
    
    print(rmse_list)
        

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

















