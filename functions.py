import pandas as pd
import numpy as np

from tqdm import tqdm
from datetime import date

def get_stop_intervals(df:pd.DataFrame) -> pd.DataFrame:
    """
        TO DO:
            1. Maybe? Change clause orders so that we don't skip nested stop intervals

		This function is designed to create a new DataFrame fron WeMob data, adding a row everytime a stop is detected. A stop is 
		detected here when:

			CASE 1. A truck stops sending CANBUS data for more than 1 minute (this is the update period of the sensor)
			CASE 2. A truck stops (speed=0 & rpm=0) or stays in slow motion (speed=0 & rpm>0) for more than 1 minute (more than a data
					row with one of these status is received)

		The data stored in each row is:
			1. Truck plate
			2. status (stop/slow_motion)
			3. latitude
			4. longitude
			5. start (moment in which the truck stops)
			6. end (moment in which the truck resumes the route)
			7. delta = end - start (amount of time the truck is stopped)
			8. engine (ON -> case 2/OFF -> case 1)

		The goal that motivates this function is to caracterize stop intervals (delta) in order to better understand the different
		scenarios that derive from a truck stopping. In simple words, it is useful for studying mean stopping times, modes etc...
	"""
    # If you are using raw data (simply as extracted from the database), there are some things you need to calculate first
    df = split_fractal_into_columns(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Adding truck motion status (the name of the function will surprise you)
    df = add_status(df)

    df = add_prev_values(df)

    # This is the DataFrame that we are going to fill and return
    intervals = pd.DataFrame(columns=["plate", "status", "begin_driver", "end_driver", "begin_lat", "begin_long", "end_lat", "end_long", "start", "end", "delta"])
    # This variable will store the difference in index between beggining and end of stop interval
    di = 0
    # Minimum speed threshold for detecting missing data
    SPEED_THRESHOLD = 50
    # Iterate through list of truck plates
    for truck, sample in tqdm(df.groupby("plate")):
        # last_index stores the index of the last end of stop recorded
        last_index = -1
        sample.reset_index(drop=True, inplace=True)
        print("-------TRUCK  " + str(truck) + "--------")
        # Iterate through filtered dataframe
        for _, row in tqdm(sample.iterrows(), total=sample.shape[0]):
            time1 = row["timestamp"]
            driver1 = row["driver"]
            # This prevents the code from register the same stop interval more than once (if a stop interval is found it will skip
            # the rows in which the stop is taking place)
            idd = row.iddata_input
            if idd <= last_index:
                continue
            try:
                intervals, last_index = check_missing_data_truck_off(intervals, last_index, df, row, SPEED_THRESHOLD)
            except Exception as e:
                print(e)
            # Now we check if the truck is stopped or in slow motion (CASE 2)
            if idd <= last_index:
                continue        
            if row["vehicle_status"] == "stop" or row["vehicle_status"] == "slow motion":
                # print("stop detected at index " + str(index))
                # We are going to find the next row in the table in which the truck's status is "on going". This will give us the time
                # at which this truck resumes its way.
                try:
                    end_row = sample[(sample.vehicle_status != row.vehicle_status) & (sample.iddata_input > idd)].iloc[0]
                # We may not find this (we stop receiving data with the truck in "stop" or "slow motion" status)
                except:
                    continue
                if end_row.empty:
                    continue
                time2 = end_row.timestamp
                # We save the index in order to skip the CANBUS data stored in between (we know that the status will be "stop" or
                # "slow motion" so it belongs to the same stop interval)
                last_index = end_row.iddata_input
                delta = (time2 - time1).total_seconds()/60
                # Let's check if we are missing data in the rows we have skipped
                middle_rows = sample[(sample.timestamp>=time1) & (sample.timestamp<=time2)]
                if len(middle_rows)-1 != delta:
                    # If there is less data rows than the length of the stop intervals that means we have skipped some data (we receive a row each minute), so we must
                    # check it
                    for _,row in middle_rows.iterrows():
                        intervals, last_index = check_missing_data_truck_off(intervals, last_index, df, row, SPEED_THRESHOLD)
                    continue
                intervals = add_interval(intervals, row, end_row, delta)
                """
                di = 0
                rem = sample.shape[0] - index
                while di < rem - 1:
                    # print(str(di) + " < " + str(rem - 1) )
                    if sample.iloc[index + di]["vehicle_status"] == "on going":
                        break
                    di = di + 1
                # print("length: ", str(sample.shape[0]))
                # print("di: ", str(di))
                if di == rem - 1:
                    time2 = time1
                    delta = -1
                else:
                    time2 = sample.iloc[index + di].timestamp
                    last_index = index + di
                    delta = (time2 - time1).total_seconds()
                    row["engine"] = "ON"
                    intervals = add_interval(intervals, row, time1, time2, delta)
                """
    return intervals

def check_missing_data_truck_off(intervals, last_index, df, row, SPEED_THRESHOLD):
    # This function checks when the last data from this truck was received. If it was received more than 1 minute ago it
    # will register a stop interval
    time1 = row["timestamp"]
    # Check timestamp of previous received data (here we use the id from the CANBUS data itself (iddata_input) in order
    # to assure that we are not skipping data)
    try: 
        prev_time_row = df[df.iddata_input == (row.iddata_input - 1)]
    except:
        return intervals, last_index
    # We need to check if a previous record exists
    if not(prev_time_row.empty):
        prev_time = prev_time_row["timestamp"].values[0]
        # Compute time delta between actual row and previous row in seconds
        delta = (np.datetime64(time1) - prev_time)/ np.timedelta64(60, 's')
        # If it's longer than a minute we register a stop interval
        if delta > 1:
            print(prev_time, time1)
            if (prev_time_row["speed"].values[0] > SPEED_THRESHOLD) and (row["speed"] > SPEED_THRESHOLD):
                prev_time_row["vehicle_status"] = "missing data"
                print("missing data detected")
                print(prev_time_row.iloc[0])
            # Here we are in CASE 1. The truck engine has stopped and we didn't receive any data.
            else:
                prev_time_row["vehicle_status"] = "truck off"
                print("truck off detected")
            # A stop interval is registered, the index of the data we are checking is stored in last_index so we can resume
            # iteration here and we continue
            intervals = add_interval(intervals, prev_time_row.iloc[0], row, delta)
            last_index = row.iddata_input
    return intervals, last_index

def add_interval(intervals, row, end_row, delta):
    ser = pd.Series(data={
        "plate": row.plate,
        "status": row["vehicle_status"],
        "begin_driver": row.driver,
        "end_driver": end_row.driver,
        "begin_lat": row.latitude,
        "begin_long": row.longitude,
        "end_lat": end_row.latitude,
        "end_long": end_row.longitude,
        "start": row.timestamp,
        "end": end_row.timestamp,
        "delta": delta,
        #"engine": row.engine
    }, index=["plate", "status", "begin_driver", "end_driver", "begin_lat", "begin_long", "end_lat", "end_long", "start", "end", "delta"])
                # print(ser)
    return intervals.append(ser, ignore_index=True)
                
                # print("delta: " + str((time2 - time1).total_seconds()))


#----------------AUX FUNCTIONS--------------------------

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

def add_status(df:pd.DataFrame) -> pd.DataFrame:
 	# Adding truck motion status (slow_motion, stop, on going)
    
    conditionList = [
    (df['speed'] == 0) & (df['rpm'] == 0),
    (df['speed'] == 0) & (df['rpm'] > 0),
    (df['speed'] > 0) & (df['rpm'] > 0)]
    choiceList = ['stop', 'slow motion', 'on going']

    df['vehicle_status'] = np.select(conditionList, choiceList, default='Not Specified')

    return df

def add_prev_values(df:pd.DataFrame) -> pd.DataFrame:
	"""
		Adds new columns to the dataframe including previous values. Useful for checking deltas, changes etc...
	"""
	# Create previous time
	df['previous_timestamp'] = df['timestamp'].shift()
	# Create previous driver
	df['previous_driver'] = df['driver'].shift()
	df['previous_status'] = df['vehicle_status'].shift()
	df["prev_plate"] = df.plate.shift()

	
	df['previous_timestamp'].fillna(pd.to_datetime('2000-01-01T00:00:00.000Z'), inplace=True)
	df['previous_driver'].fillna('--', inplace=True)
	df['previous_status'].fillna('Not Specified', inplace=True)
	df.reset_index(drop=True, inplace=True)

	return df
