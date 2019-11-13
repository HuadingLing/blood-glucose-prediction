#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run an experiment."""

import logging
import sys
import os
import yaml
#import pprint
import importlib.util
import tensorflow as tf
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import numpy as np
import metrics
import visualization as vs

LSTM_states = [8,16,32,64,96,128,256,384,516]
n_past = [5,10,15,20,25,30] #长度<=7

#LSTM_states = [8,16]
#n_past = [5,10] #长度<=7

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

    #mse_result = np.zeros((len(n_past),len(LSTM_states)))
    mse_result = []
    for past_steps in n_past:
        x_train, y_train, x_valid, y_valid, x_test, y_test = module_dataset.load_glucose_dataset(
            xlsx_path = cfg['dataset']['xlsx_path'],
            nb_past_steps   = past_steps,
            nb_future_steps = int(cfg['dataset']['nb_future_steps']),
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
        for states in LSTM_states:
            np.random.seed(seed)
        
            # training mode
            if mode == 'train':
                print("loading optimizer ...")
                optimizer = module_optimizer.load(
                    float(cfg['optimizer']['learning_rate'])
                )
        
                print("loading loss function ...")
                loss_function = module_loss_function.load()
                print("loaded function {} ...".format(loss_function.__name__))
        
                print("loading model ...")
                if loss_function.__name__ == 'tf_nll':
                    model = module_model.load_with_lstm_states(
                        x_train.shape[1:],
                        y_train.shape[1]*2,
                        states
                    )
                else:
                    model = module_model.load_with_lstm_states(
                        x_train.shape[1:],
                        y_train.shape[1],
                        states
                    )
        
                model.compile(
                    optimizer=optimizer,
                    loss=loss_function
                )
        
                print(model.summary())
        
                print("training model ...")
                model = module_train.train(
                    model          = model,
                    x_train        = x_train,
                    y_train        = y_train,
                    x_valid        = x_valid,
                    y_valid        = y_valid,
                    batch_size     = int(cfg['train']['batch_size']),
                    epochs         = int(cfg['train']['epochs']),
                    patience       = int(cfg['train']['patience']),
                    shuffle        = cfg['train']['shuffle'],
                    artifacts_path = cfg['train']['artifacts_path']
                )

                y_pred_on_valid = model.predict(x_valid)
                y_pred_on_valid_last = y_pred_on_valid[:,-1].flatten()/scale
                y_valid_last = y_valid[:,-1].flatten()/scale
                
                del model # 删除已训练完模型
                
                mse = metrics.root_mean_squared_error(y_pred_on_valid_last, y_valid_last)
                mse_result.append(mse)
                
    output_image_dir = "output_image/search_state_and_past_steps/patient_"+str(cfg['dataset']['patient_id'])+"/"
    if os.path.exists(output_image_dir) == False:
        os.makedirs(output_image_dir)
        
    save_file_name_prefix = output_image_dir + "sheet_" + str(cfg['dataset']['sheet_pos'])+"_loss_function_"+loss_function.__name__+"_"
        
    vs.plot_mse(mse_result, LSTM_states, n_past, save_file_name_prefix + '.png')
    print(mse_result)
        

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
