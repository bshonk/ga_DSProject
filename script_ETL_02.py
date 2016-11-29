# ETL pipeline for sample bike data
# build dictionary of counts at anygiven time
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import sys


station_ids = pd.read_csv('data_clean/all_station_ids.csv', header=None, names=['id'])
list_ids = [i for i in set(station_ids.id.tolist())]

# run this code at the same time for different ranges (up to 535)
start = sys.arg[1]
stop = sys.arg[2]

list_ids = list_ids[start:stop] 

# connect to occupancy data
occupancy_all = pd.read_csv('data_clean/occupancy.csv', header=0, index_col='arrival_time', parse_dates=True)

for i in list_ids:
	startTime = datetime.now()
	station_id = i
	occupancy = occupancy_all[occupancy_all.station_id == station_id]

	# create a range of all possible timestamps
	min_dt = occupancy.index.min()
	max_dt = max(occupancy.index + pd.to_timedelta(occupancy.duration, unit='m'))
	dt_range = int((max_dt - min_dt).total_seconds() / 60) + 1

	# create a list of all possible timestamps
	dt_list = [min_dt + timedelta(minutes=i) for i in range(dt_range)]
	zeros = [0] * dt_range

	# turn that the list of all possible timestamps into an empty dictionary
	occupancy_counts = dict(zip(dt_list,zeros))

	# delete list of dates
	del dt_list, zeros

	# dictionary of counts per minute at the given station
	for index, row in occupancy.iterrows():
	    for i in range(row['duration']):
	       master_time = index + timedelta(minutes=i)
	       occupancy_counts[master_time] += 1

	del occupancy

	# convert dictionary to df
	occupancy_minute = pd.DataFrame(occupancy_counts.items(), columns=['date_time','count'])

	# delte dictionary
	del occupancy_counts

	# set index
	occupancy_minute = occupancy_minute.set_index('date_time')

	# import dockcount data
	dpcount = pd.read_csv('data_clean/all_stations.csv', header=0)
	dpcount = dpcount[dpcount.Id == station_id].reset_index(drop=True)

	dock_count = dpcount.Dpcapacity.min()

	# flag full and empty mintues
	occupancy_minute['empty'] = occupancy_minute['count'] == 0
	occupancy_minute['full'] = occupancy_minute['count'] >= dock_count

	# round mintues to 30 min (round down ie 6:59 goes to 6:30)
	# group by 30 min periods
	occupancy_minute = occupancy_minute.resample('30T').agg({'count':['mean','min','max'], 'empty':'sum', 'full':'sum'})

	# rename the columns
	occupancy_minute.columns = [' '.join(col).strip() for col in occupancy_minute.columns.values]

	# rename columns
	columns = ['mean_count','min_count','max_count', 'minutes_full', 'minutes_empty']
	occupancy_minute.columns = columns

	# write to csv
	output_file = str(station_id) + '.csv'
	occupancy_minute.to_csv('data_output/' + output_file)

	print datetime.now() - startTime

