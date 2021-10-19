import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import openpyxl
# from pylab import *
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import time
import csv
import sklearn
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARIMA  
# import forecast
import torch
from torch import nn
from sklearn import metrics

Work_PATH = 'D:\\workplace\\partime\\LiuZiyan\\MSc_Project\\workplace\\'
Data_PATH = Work_PATH + 'Shanghai-Telcome-Six-Months-DataSet'
Figure_PATH = Work_PATH +'Figure\\'
psdata_xlsx_PATH = Work_PATH + 'psdata.xlsx' 
psdata_csv_PATH = Work_PATH + 'psdata.csv' 

original_date_str = '2014-06-01 00:00:00'
original_date = datetime.strptime(original_date_str, '%Y-%m-%d %H:%M:%S')

ds = {'month': 0, 'date': 1, 'start_time': 2, 'end_time': 3, 'latitude': 4, 'longitude': 5, 'user_id': 6}

class time_series_map(object):
	def __init__(self):
		super(time_series_map, self).__init__()
		# self.Data_PATH = Data_PATH
		# self.period_ratio = period_ratio
		self.sec_data = self.init_sec_data(183)
		self.min_data = []
		self.period_data = []

	def init_sec_data(self, days):
		ts_day = []
		for i in range(0, days*1440*60):
			ts_day.append(0)
		return ts_day

	def read_data_xlsx(self, file_PATH):
		excel_file = openpyxl.load_workbook(filename = file_PATH, read_only = True)
		data = []
		row_num = -1
		for sheet in excel_file:
			for row in sheet.rows:
				if row_num == -1:
					row_num = 0
					continue
				data.append([])
				for cell in row:
					data[row_num].append(cell.value)
				row_num += 1
				# if row_num > 62:
				# 	break
		for row in data:
			start_t = row[ds['start_time']]
			end_t = row[ds['end_time']]
			if start_t < original_date:
				start_t = original_date
			start_sec = int((start_t - original_date).total_seconds())
			end_sec = int((end_t - original_date).total_seconds())
			for i_sec in range(start_sec, end_sec+1):
				self.sec_data[i_sec] = self.sec_data[i_sec] + 1

	def read_datas_xlsx(self, Data_PATH):
		filenames = os.listdir(Data_PATH)
		for filename in filenames:	
			self.read_data_xlsx(Data_PATH +'\\'+ filename)
			# self.read_data_xlsx(Data_PATH +'\\'+ 'data_6.1~6.15.xlsx')
			# break

	def period_change(self):
		for i in range(0, len(self.sec_data)):
			if (i%60) == 0:
				self.min_data.append(self.sec_data[i])
			if (i%(60*15)) == 0:
				self.period_data.append(self.sec_data[i])

	def init_period_change(self, period_ratio):
		data = []
		for i in range(0, len(self.sec_data)):
			if (i%period_ratio) == 0:
				data.append(self.sec_data[i])
		return data

	def write_to_xlsx(self, xlsx_PATH):
		wb = openpyxl.load_workbook(xlsx_PATH)
		ws = wb['Sheet1']
		ws.append(self.sec_data)
		ws.append(self.min_data)
		ws.append(self.period_data)
		wb.save(xlsx_PATH)	

	def write_to_csv(self, xlsx_PATH):
		with open(xlsx_PATH,'w') as f:
		    f_csv = csv.writer(f)
		    f_csv.writerow(self.sec_data)
		    f_csv.writerow(self.min_data)
		    f_csv.writerow(self.period_data)

	def read_datas_csv(self, Data_PATH):	
		with open(Data_PATH) as f:
		    f_csv = csv.reader(f)
		    sign = 0
		    for row in f_csv:
        		if sign == 0:
        			for i in range(0,len(row)):
        				self.sec_data.append(int(row[i]))
        			sign += 1
        		elif sign == 1:
        			for i in range(0,len(row)):
        				self.min_data.append(int(row[i]))
        			sign += 1
        		elif sign == 2:
        			for i in range(0,len(row)):
        				self.period_data.append(int(row[i]))
        			sign += 1

if __name__ == '__main__':
	ts = time_series_map()
	# ts.read_datas_xlsx(Data_PATH)
	# ts.period_change()
	# ts.write_to_xlsx(psdata_xlsx_PATH)
	# ts.write_to_csv(psdata_csv_PATH)
	# ts.period_data = ts.init_period_change(60*15)

	ts.read_datas_csv(psdata_csv_PATH)
	time = []
	small_data = []
	for i in range(0, len(ts.period_data)):
	# for i in range(0, 50*96):
		time.append(i)
		small_data.append(ts.period_data[i])

	#归一化
	small_data = np.array(small_data)
	max_v = np.max(small_data)
	# print(max_v)
	min_v = np.min(small_data)

	sd = []
	# for i in range(0, 50*96):
	for i in range(0, len(ts.period_data)):
		sd.append((small_data[i]-min_v)/(max_v-min_v))
	small_data = sd