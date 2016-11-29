
import pandas as pd
import os
import random
from datetime import timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np

files = os.listdir('data_output/')
files = [i for i in files if 'csv' in i]
#files = random.sample(files,10)
#files = ['409.csv','410.csv']

# empty list for confusion scores
# set up the csv for actuals data
output = []
header = pd.DataFrame(columns=['date_time','mean_count', 'y', 'y_pred', 'station_id', 'split'])
header.to_csv('data_decisiontree/deciscion_tree_predictions2.csv', mode='a', index=False, header=True)

for station in files:
	station_id = station
	url = 'data_output/'
	data = pd.read_csv(url+str(station_id), index_col='date_time', parse_dates=True)

	# self join data set offset on hours
	time_offset = 4
	feature = pd.read_csv(url+str(station_id), index_col='date_time', parse_dates=True)
	feature.index = feature.index + timedelta(minutes=(time_offset*30))
	columns = [str(4*30)+'_'+ name for name in feature.columns]
	feature.columns = columns
	data_feature = pd.concat([data,feature], axis=1, join='inner')


	# y class
	data_feature['minutes_empty'] = (data_feature.minutes_empty > 0).astype(int)

	# X and y
	feature_cols = [ i for i in data_feature.columns[2:] if 'full' not in i and 'empty' not in i and str(time_offset*30) in i]
	X = data_feature[feature_cols]
	y = data_feature[['minutes_empty']]



	# only run the model if it meets a .5% threshold of events
	if (y.minutes_empty > 0).sum() / (len(y) * 1.0) < .01:
		print 'nothing'
	else:
		print station_id + 'model'

		# train test split (75% train, 25% test)
		# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
		
		percent = .5
		rows = int(percent * len(X)) 
		rows_test = len(X) - rows
		X_train = X.head(rows)
		X_test = X.tail(rows_test)
		y_train = y.head(rows)
		y_test = y.tail(rows_test)

		if y_test.minutes_empty.sum() == 0:
			print station_id + 'not events in test'
		else:

			# tree classification fit
			treeclf = DecisionTreeClassifier(max_depth = 3, random_state=1)
			treeclf.fit(X_train, y_train)

			# predictions
			y_pred = treeclf.predict(X_test)
			confusion = metrics.confusion_matrix(y_test, y_pred)
			# columns = ['TN','FN', 'FP', 'TP']
			confusion = confusion[0].tolist() + confusion[1].tolist()

			# add rows to final
			row = [int(station_id[:station_id.find('.csv')])] + confusion
			output.append(row)

			# y actual, y predicted, mean count actuals
			y_pred_all = treeclf.predict(X)
			y_pred_df = pd.DataFrame(y_pred_all, index=y.index, columns=['y_pred'])
			y.columns = ['y']
			mean_count = data[['mean_count']]

			# output df
			output_actuals = pd.concat([mean_count, y, y_pred_df], axis=1)
			output_actuals['station_id'] = int(station_id[:station_id.find('.csv')])

			# label as train or test
			y_train['split'] = 'train'
			y_test['split'] = 'test'
			test_train = pd.concat([y_train, y_test], axis=0).sort_index()
			output_actuals = pd.concat( [output_actuals, test_train[['split']]], axis=1 )

			output_actuals.to_csv('data_decisiontree/deciscion_tree_predictions2.csv',index=True, mode='a', header=False)



# convert to DF and output to csv
output = pd.DataFrame(output, columns=['station_id', 'TN','FN', 'FP', 'TP'])
output.to_csv('data_decisiontree/deciscion_tree2.csv',index=False)
