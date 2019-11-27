#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run an experiment."""

import logging
import sys
import os
import yaml

import importlib.util

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import numpy as np
import matplotlib.pyplot as plt
import visualization as vs

true_id=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,27,29,30,31,32,33,35,37,38,39,40,41,43,44,45,47,48,50]

patient_num = 41
sheet_num = 13

def main(yaml_filepath, mode):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    module_dataset = load_module(cfg['dataset']['script_path'])
    for i in range(patient_num):
        for sheet_pos in range(sheet_num):
            patient_id = i + 1
            patient_true_id = true_id[patient_id-1]
            df_glucose_level =  module_dataset.load_ohio_series(xlsx_path = cfg['dataset']['xlsx_path'],
                                                                sheet_pos = sheet_pos, 
                                                                patient_id = patient_id)
            output_image_dir = "output_image/ID_search/patient_" + str(patient_true_id) + "/"
            if os.path.exists(output_image_dir) == False:
                os.makedirs(output_image_dir)
            nd_glucose_level = df_glucose_level.values
            plt.plot(nd_glucose_level)
            plt.grid()
            plt.xlabel('Time (hours)')
            plt.ylabel('Glucose Level (mg/dL)')
            plt.title('Patient '+str(patient_true_id)+'  '+'Sheet '+str(sheet_pos))
            plt.savefig(output_image_dir+'sheet_ '+str(sheet_pos)+'.png')
            #supported formats: eps, pdf, pgf, png, ps, raw, rgba, svg, svgz
            plt.show()
    

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


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)
    parser.add_argument("-m", "--mode",
                        dest="mode",
                        help="mode of run",
                        metavar="FILE",
                        required=True)
    return parser


if __name__ == '__main__':
    #args = get_parser().parse_args()
    #main(args.filename, args.mode)
    filenames = 'experiments/example.yaml'
    mode = 'train'
    #mode = 'evaluate'
    main(filenames, mode)
