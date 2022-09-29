import pandas as pd
from functions import *

def _main_():
    toy_df = pd.read_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/db_1month_9trucks.csv')
    print('CSV loaded successfully')
    # print(toy_df)

    intervals = get_stop_intervals(toy_df)
    intervals.to_csv('intervals_1month_9trucks.csv')
    print('intervals saved in same folder')
    print('EOS')

if __name__ == "__main__":
    _main_()