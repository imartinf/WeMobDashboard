"""
    Data extraction, preparation and visualization from WeMob DB
    Iván Martín Fernández
    This script serves as a first approach to access the MySQL database that stores CANBUS data from trucks using Python. 
    It connects to the database and performs a first data cleansing.
    It also creates a dashboard, based on a previous work performed by Hengxuan Ying in his EoD Project, to visualize the data. Access to the database provided by WeMob.
    Comments in Spanish/English, sorry for the inconvenience :)

    Usage:
        python run.py <conf> [--options]

    Options:
        --limit Number of rows to load from database. Useful for debugging
        -h Display this message

"""
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../../')

import argparse
import json
import numpy
import os
from datetime import datetime

import app
from preprocessing import preprocessing as prep
from connection import get_db
from app import create_and_run_app

def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description='Connection to WeMob DB, preprocessing and visualization dashboard',
        formatter_class=argparse.RawTextHelpFormatter)

    # Mandatory arguments
    parser.add_argument("conf",
        type=str,
        help="Path to .json file with configuration parameters for SQL server connection")

    # Options
    parser.add_argument("--limit",
        type=int,
        help="Number of rows to load from database. Useful for debugging")
    return parser.parse_args()


def display_params(args, params):
    """
        Display configuration paramenters on terminal
    """
    print("SCRIPT: " + os.path.basename(__file__))
    print('Options...')
    for arg in vars(args):
        print('  ' + arg + ': ' + str(getattr(args, arg)))
    print('-' * 30)

    print('Config-file params...')
    for key, value in params.items():
        print('  ' + key + ': ' + str(value))
    print('-' * 30)

def load_json(json_file):
    """
        Load parameters from a configuration json file
    """
    with open(json_file, 'r') as file:
        jdata = json.load(file)
    return jdata

def _main_():
    args = parse_command_line_args()
    params = load_json(args.conf)
    display_params(args, params)
    init_time = datetime.now()
    print("Process initiated at time: ", init_time.strftime("%H:%M:%S"))

    db = get_db(params, args)
    df = prep(db)

    create_and_run_app(df)

    print("Finished at time: ", datetime.now().strftime("%H:%M:%S"), "Execution time: ", str(datetime.now() - init_time))

if __name__ == "__main__":
    _main_()



