import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import openpyxl
# from pylab import *
import numpy as np
import matplotlib; matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import time
# import torch
# x = torch.rand(5, 3)
# print(x)

Work_PATH = 'D:\\workplace\\partime\\LiuZiyan\\MSc_Project\\workplace\\submit_v0.1\\'
Data_PATH = Work_PATH
Figure_PATH = Work_PATH +'Figure\\'

# filenames = os.listdir(Data_PATH)
# for filename in filenames:
# 	excel_file = openpyxl.load_workbook(Data_PATH +'\\'+ filename)


class time_series_map(object):
	"""docstring for time_series_map"""

	def __init__(self):
		super(time_series_map, self).__init__()
		self.day = []
		self.stations = []
		# self.a_day = []

	def init_ts_day(self):
		ts_day = []
		for i in range(0, 1440*60):
			ts_day.append(0)
		return ts_day

original_date_str = '2014-06-01 00:00:00'
original_date = datetime.strptime(original_date_str, '%Y-%m-%d %H:%M:%S')
# od_t = time.mktime(original_date.timetuple())
# print(datetime_object)
ds = {'month': 0, 'date': 1, 'start_time': 2, 'end_time': 3, 'latitude': 4, 'longitude': 5, 'user_id': 6}

if __name__ == '__main__':
	excel_file = openpyxl.load_workbook(Data_PATH  + 'day_6.1.xlsx')
	ts_map = time_series_map()
	# ts_map.days.append(ts_map.init_ts_day())

	# ctr_sign = 0
	data = []
	# position_sign = 0
	row_num = 0
	for sheet in excel_file:
		for row in sheet.rows:
			data.append([])
			for cell in row:
				# print(cell.value)
				data[row_num].append(cell.value)
			row_num += 1
			# position_sign += 1
			# ctr_sign += 1
			# if ctr_sign	== 2:
			# 	break
	for row in data:
		if row[ds['month']] == 201406:
			if row[ds['date']] == 1:
				if row[ds['latitude']] and row[ds['longitude']]:
					sign = False
					for st in ts_map.stations:
						if (st[0] == row[ds['latitude']]) and (st[1] == row[ds['longitude']]):
							sign = True
							start_t = row[ds['start_time']]
							end_t = row[ds['end_time']]
							if start_t < original_date:
								start_t = original_date
							start_min = int((start_t - original_date).total_seconds())
							end_min = int((end_t - original_date).total_seconds())
							# start_t = time.mktime(row[ds['start_time']].timetuple())
							# if start_t < od_t:
							# 	start_t = od_t
							# end_t = time.mktime(row[ds['end_time']].timetuple())
							# start_min = int((start_t - od_t)/60)
							# end_min = int((end_t - od_t)/60)
							for i_min in range(start_min, end_min+1):
								ts_map.day[st[2]][i_min] += 1
							break
					if sign == False:
						len_stations = len(ts_map.stations)
						ts_map.stations.append([row[ds['latitude']], row[ds['longitude']], len_stations])
						ts_map.day.append(ts_map.init_ts_day())
						# print(len_stations)
						start_t = row[ds['start_time']]
						end_t = row[ds['end_time']]
						if start_t < original_date:
							start_t = original_date
						start_min = int((start_t - original_date).total_seconds())
						end_min = int((end_t - original_date).total_seconds())
						# print(start_min,end_min)
						# start_t = time.mktime(row[ds['start_time']].timetuple())
						# if start_t < od_t:
						# 	start_t = od_t
						# end_t = time.mktime(row[ds['end_time']].timetuple())
						# start_min = int((start_t - od_t)/60)
						# end_min = int((end_t - od_t)/60)
						for i_min in range(start_min, end_min+1):
							ts_map.day[len_stations][i_min] += 1
	x_axis = []
	y_axis = []
	for i in range(0, 1440):
		x_axis.append(i)
		# print(ts_map.day[0][i])
		# y_axis.append(ts_map.day[0][i*60])
	for st in ts_map.stations:
		y_axis = []
		for i in range(0, 1440):
			y_axis.append(ts_map.day[st[2]][i*60])
		plt.plot(x_axis, y_axis)
		plt.title("Location: "+str(ts_map.stations[0])+","+str(ts_map.stations[1])) # 图形标题
		plt.xlabel("Minutes of a day.") # x轴名称
		plt.ylabel("Load (Connections) of the device at the location.") # y 轴名称
		str_figname = Figure_PATH + 'Fig_' + str(st[2]) + '.png'
		plt.savefig(str_figname)
		plt.clf()
	# plt.plot(x_axis, y_axis)
	# plt.title("Station: "+str(ts_map.stations[0])+","+str(ts_map.stations[1])) # 图形标题
	# plt.xlabel("Minutes of a day.") # x轴名称
	# plt.ylabel("Load of the station.") # y 轴名称
	# plt.show()
	print("plot complete")

						

					
		