import logging
import sys
import os
import yaml
import importlib.util

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from datasets.ohio import load_glucose_dataset
from loss_functions.nll_keras import tf_nll

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
sheet_num = 13

batch_size = 256
epochs = 10

past_steps = 12
lstm_states = 128


use_nll = False

#True_id = [5,9,11,12]

def main(yaml_filepath, mode):
    if use_nll == True:
        output_shape = 2
        loss = tf_nll
        loss_name = 'nll'
    else:
        output_shape = 1
        loss = 'mean_squared_error'
        loss_name = 'mse'
    id_list = []
    rmse_list = []
    for pid in [11]:
        patient_true_id = true_id[pid-1]
        id_list.append(patient_true_id)
        output_image_dir = "output_image/table/LSTM_single_sheet/patient_"+str(patient_true_id)+"/"
        if os.path.exists(output_image_dir) == False:
            os.makedirs(output_image_dir)
                    
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_glucose_dataset(
                xlsx_path = xlsx_path,
                nb_past_steps   = past_steps,
                nb_future_steps = future_steps,
                max_length = 256,
                train_fraction  = 0.6,
                valid_fraction  = 0.2,
                test_fraction   = 0.2,
                sheet_pos = sheet_num - 1,
                patient_id = pid,
                )
        x_train *= scale
        y_train *= scale
        x_valid *= scale
        y_valid *= scale
        x_test *= scale
        y_test *= scale
        
        np.random.seed(seed)

        sequence_input = Input(shape=(past_steps, feature))
        
        lstm_feture_without_dropout = LSTM(lstm_states)(sequence_input)
        lstm_feture = Dropout(0.1)(lstm_feture_without_dropout)
        pred_output = Dense(output_shape)(lstm_feture)
        predition_model = Model(inputs=sequence_input, outputs=pred_output)
        
        predition_model.compile(optimizer='adam', loss=loss)
        
        loss_1_train = predition_model.evaluate(x_train, y_train, batch_size=batch_size)
        loss_1_valid = predition_model.evaluate(x_valid, y_valid, batch_size=batch_size)
        
        predition_model.fit(x_train, y_train, 
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_valid, y_valid),
                            callbacks=[EarlyStopping(patience=10)])
        
        loss_2_train = predition_model.evaluate(x_train, y_train, batch_size=batch_size)
        loss_2_valid = predition_model.evaluate(x_valid, y_valid, batch_size=batch_size)

        del predition_model
        
        print([loss_1_train, loss_1_valid, loss_2_train, loss_2_valid])
        
        

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

















