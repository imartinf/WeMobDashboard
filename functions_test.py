import pandas as pd
from functions import *

def _main_():
    toy_df = pd.read_csv('../3306JZR_limit1000.csv')
    print('CSV load successfully')
    # print(toy_df)

    intervals = get_stop_intervals(toy_df)
    intervals.to_csv('intervals_v2.csv')
    print('intervals saved in same folder')
    print('EOS')

if __name__ == "__main__":
    _main_()