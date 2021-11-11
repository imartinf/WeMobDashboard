"""
	Data preprocessing and cleaning as seen in previous works. Full pipeline is broken down into small functions in order to make it modular and easier to change.
"""

import pandas as pd
import numpy as np

def print_info(df:pd.DataFrame):
	"""
		Prints some usefull information
	"""
	print('-'*30)
	print(df.info())
	print(df.describe())

def clean(df:pd.DataFrame) -> pd.DataFrame:
	"""
		Performs basic cleaning to database
	"""
	# Remove duplicate rows (there are usually no duplicates, but jic)
	df = df.drop_duplicates()
	# Timestamp as datetime
	df.timestamp = pd.to_datetime(df.timestamp)
	# Check 0<fuel_level<100
	df.fuel_level[(df.fuel_level<0) | (df.fuel_level>100)] = np.nan
	return df


# Clean plate strings (they must all be a combination of letters and numbers without spaces so we eliminate them
# and everything that goes after them)

# This is designed ad hoc for some errors that are of the type: 5498LFN ITER FRUGAL
# COMMENTED: DO NOT CLEAN PLATES AS THEY ARE IDs DEFINED BY THE CLIENTS
# df.plate = df.plate.str[:7]

# TO DO: Check if acumulative data is indeed acumulative. This code block is almost the same as Heng's (incomplete)
# ac = ["engine_hours", "total_consum", "accelerations", "breaks", "clutch", "slowmotion_time"]
# dfac = df[ac]
# same = [df.iloc[i].plate == df.iloc[i-1].plate for i in df.index]
# same

def split_fractal_into_columns(df:pd.DataFrame) -> pd.DataFrame:
	"""
		Name is self-explainatory
	"""
	# Split fractal into different strings
	add = df.fractal.str.split(',', expand=True)
	add.describe()

	# Columns 2 and 3 are always empty
	add = add.drop(columns=[2, 3])

	# Assign split values and drop original
	df["street"] = add[0]
	df["postal_code"] = add[1]
	df["city"] = add[4]
	df["province"] = add[5]
	df["state"] = add[6]
	df["country"] = add[7]

	return df.drop(columns=["fractal"])

def preprocessing(df:pd.DataFrame) -> pd.DataFrame:
	"""
		Execute every function in this module
	"""
	df = clean(df)
	df = split_fractal_into_columns(df)
	print_info(df)
	return df
