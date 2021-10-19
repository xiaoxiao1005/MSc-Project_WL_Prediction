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

INPUT_FEATURES_NUM = 1
OUTPUT_FEATURES_NUM = 1
BATCH_NUM = 20

class time_series_map(object):
	def __init__(self):
		super(time_series_map, self).__init__()
		# self.Data_PATH = Data_PATH
		# self.period_ratio = period_ratio
		self.sec_data = self.init_sec_data(183)
		self.min_data = []
		self.period_data = []	
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


class train_test_data(object):
	def __init__(self, train_ratio, ts_data):
		self.train_ratio = train_ratio
		self.train_data = self.init_split(0, int(train_ratio*len(ts_data)), ts_data)
		self.test_data = self.init_split(int(train_ratio*len(ts_data)), len(ts_data), ts_data)

	def init_split(self, lb, hb, data):
		sp_data = []
		for i in range(lb, hb):
			sp_data.append(data[i])
		return sp_data

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
	"""
		Parameters：
		- input_size: feature size
		- hidden_size: number of hidden units
		- output_size: number of output
		- num_layers: layers of LSTM to stack
	"""
	def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
		super().__init__()
 
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
		self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
	def forward(self, _x):
		_x = _x.view(len(_x), BATCH_NUM, 1)
		# print(_x)
		# _x = _x.squeeze(-1)
		x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
		s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
		x = x.view(s*b, h)
		x = self.forwardCalculation(x)
		# print(x)
		x = x.view(s, b, -1)
		# output = []
		# print(x)
		# for el in x:
		# 	output.append(el[-1][-1])
		# output = torch.tensor(output)
		# print(output)
		x = x.narrow(1, 0, 1)
		x = x.squeeze(-1)
		x = x.squeeze(-1)
		# print(x)
		return x

class GRUnn(nn.Module):
	"""
		Parameters：
		- input_size: feature size
		- hidden_size: number of hidden units
		- output_size: number of output
		- num_layers: layers of LSTM to stack
	"""
	def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
		super().__init__()
 
		self.gru = nn.GRU(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
		self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
	def forward(self, _x):
		_x = _x.view(len(_x), BATCH_NUM, 1)
		# print(_x)
		# _x = _x.squeeze(-1)
		x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
		s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
		x = x.view(s*b, h)
		x = self.forwardCalculation(x)
		# print(x)
		x = x.view(s, b, -1)
		# output = []
		# print(x)
		# for el in x:
		# 	output.append(el[-1][-1])
		# output = torch.tensor(output)
		# print(output)
		x = x.narrow(1, 0, 1)
		x = x.squeeze(-1)
		x = x.squeeze(-1)
		# print(x)
		return x


def time_slice(time,single,X_lag):
    sample = []
    label = []
    for k in range(len(time) - X_lag - 1):
        t = k + X_lag
        sample.append(single[k:t])
        label.append(single[t + 1])
    return sample,label

if __name__ == '__main__':
	ts = time_series_map()
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
	#ARIMA预观察数据
	
	# scaler = preprocessing.StandardScaler().fit([small_data])
	# small_data = scaler.transform(X)[0]

	# pdata = pd.DataFrame(small_data, columns=['series'], index=time)
	# plt.plot(pdata)
	# df = pdata.diff()
	# df = df.dropna()
	# plt.plot(df)
	# df = pdata.diff(periods = 96)
	# df = df.dropna()
	# plt.plot(df)
	# df = pdata.diff(periods = 96*7)#6月9号差分6月2号，出现了峰值，由于6月2号是端午，节假日影响，6月9日是周一
	# df = df.dropna()
	# print(df)
	# plt.plot(df)
	# plt.plot(pdata)
	# plot_acf(df,lags=100)
	# plot_pacf(df,lags=100)
	# str_figname = Figure_PATH + 'Fig_' + 'diff_2*96' + '.png'
	# plt.show()

	# im = plot_acf(df,lags=1000)
	# str_figname = Figure_PATH + 'Fig_' + 'ACF_96P_HalfYear' + '.png'
	# im.savefig(str_figname)

	#切分数据生成训练和测试集
	sample,label = time_slice(time,small_data,20)
	
	X_train, X_test, y_train, y_test = train_test_split(sample, label, test_size = 0.2, shuffle = False)
	print(len(y_train),len(y_test))

	# ARIMA预测
	# print("start ARIMA")
 # 	# model = ARIMA(pdata, (3,0,16)).fit()
	# # predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)
	# # pred_outsample = model.predict(start=len(small_data)-5,end = len(small_data)+200, dynamic=True)
	# predictions_ARIMA = []
	# for i in range(0,len(X_test)):
	# 	# pdata = pd.DataFrame(X_test[i], columns=['series'], index=np.arange(len(X_test[i])))
	# 	# pdata = pdata.dropna()
	# 	model = ARIMA(X_test[i], (2,0,1)).fit()
	# 	predictions_ARIMA.append(model.forecast()[0])
	# # arma_3_16 = sm.tsa.ARMA(pdata,(8,0)).fit()
	# # predict_sunspots = arma_3_16.predict(10*96, 100*96, dynamic=True)
	# # plt.plot(pdata['series'][80*96:])
	# y_hat = predictions_ARIMA
	# # plt.plot(predictions_ARIMA,c='r')
	# # plt.plot(y_test)
	# mod_name = 'ARIMA_norm_slice100_2_0_1_bigdata'
	# y_hat = np.array(y_hat)
	# # plt.show()

	# LSTM预测
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	y_train = y_train.astype('float32')
	y_test = y_test.astype('float32')

	train_x_tensor = torch.from_numpy(X_train)
	print(train_x_tensor)
	# train_y_tensor = torch.from_numpy(train_y_tensor)
	train_y_tensor = torch.from_numpy(y_train)
	print(train_y_tensor)

	lstm_model = GRUnn(INPUT_FEATURES_NUM, hidden_size = 30, output_size = OUTPUT_FEATURES_NUM, num_layers = 1)
	print('GRU model:', lstm_model)
	# lstm_model = LstmRNN(INPUT_FEATURES_NUM, hidden_size = 60, output_size = OUTPUT_FEATURES_NUM, num_layers = 3)
	# print('LSTM model:', lstm_model)
	print('model.parameters:', lstm_model.parameters)
	loss_function = nn.MSELoss()
	optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
 
	max_epochs = 10000
	for epoch in range(max_epochs):
		output = lstm_model(train_x_tensor)
		# print(output)

		# output = output.squeeze(-1)
		# output = output.squeeze(-1)
		loss = loss_function(output, train_y_tensor)
		# print(loss)
 
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
 
		if loss.item() < 0.008:
			print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
			print("The loss value is reached")
			break
		elif (epoch+1) % 1 == 0:
			print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))

	predictive_y_for_training = lstm_model(train_x_tensor)
	predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

	test_x_tensor = torch.from_numpy(X_test)
	# print(train_x_tensor)
	# train_y_tensor = torch.from_numpy(train_y_tensor)
	test_y_tensor = torch.from_numpy(y_test)

	lstm_model = lstm_model.eval() # switch to testing model

	predictive_y_for_testing = lstm_model(test_x_tensor)
	
	loss = loss_function(predictive_y_for_testing, test_y_tensor)
	print('total Loss: {:.5f}'.format(loss.item()))
	predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
	# plt.figure()
	# plt.plot(predictive_y_for_training, 'r', label='predictive_y')
	# plt.plot(y_train, 'g', label='train_y')
	# plt.show()
	# plt.figure()
	# plt.plot(predictive_y_for_testing, 'r', label='predictive_y')
	# plt.plot(y_test, 'g', label='test_y')
	mod_name = 'GRU_norm_slice20_1_30_1_1_0.008_bigdata'
	y_hat = np.array(predictive_y_for_testing)
	# # plt.show()


	#SVR预测
	# print("start SVR")
	# svr = SVR()
	# parameters = {'kernel':['rbf'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}
	# grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=1, scoring='neg_mean_squared_error')
	# param_grid = {"C": np.linspace(10**(-2),10**3,100),'gamma': np.linspace(0.0001,1,20)}

	# mod = SVR(epsilon = 0.1,kernel='rbf')
	# grid_search = GridSearchCV(estimator = mod, param_grid = param_grid, scoring = "neg_mean_squared_error",verbose = 0)

	# grid_search = SVR(epsilon = 0.1, kernel = 'rbf', C=1e3, gamma=0.0001)
	# print("start FIT")
	# grid_search.fit(X_train,y_train)


	# mod_name = 'svr_norm_slice20_1e3_00001_bigdata'
	# mod_name = 'svr_slice20_lin'abs_vals
	# joblib.dump(grid_search, mod_name+'.pkl')
 	
	#SVR模型加载
	# grid_search = joblib.load(mod_name+'.pkl')
	 

	print("start TEST")
	# SVR模型测试
	# y_hat = grid_search.predict(X_test)

	# print(grid_search.best_params_) #显示最佳参数
	 
	# 计算预测值与实际值的残差绝对值
	# abs_vals= np.abs(y_hat-y_test)
	# mae = np.sum(abs_vals)/len(abs_vals)
	# mae = np.abs(y_hat-y_test).mean()
	# mse = ((y_hat - y_test) ** 2).mean()
	# rmse = np.sqrt(((y_hat - y_test) ** 2).mean())
	# mape = np.abs((y_hat-y_test)/y_test).mean()

	# print("MAE = ", mae)
	# print("MSE = ", mse)
	# print("RMSE = ", rmse)
	# print("MAPE = ", mape)

	MSE = metrics.mean_squared_error(y_test, y_hat)
	RMSE = metrics.mean_squared_error(y_test, y_hat)**0.5
	MAE = metrics.mean_absolute_error(y_test, y_hat)
	sum_y = 0
	print(y_hat)
	print(y_test)
	y_hat = list(y_hat)
	y_test = list(y_test)
	for i in range(0,len(y_hat)):
		sum_y += abs(y_hat[i]-y_test[i])/y_test[i]
	MAPE = sum_y/len(y_hat)
	# MAPE = np.mean(np.abs((y_hat - y_test) / y_test))
	print("MAE = ", MAE)
	print("MSE = ", MSE)
	print("RMSE = ", RMSE)
	print("MAPE = ", MAPE)
	# x_axis = []
	# y_axis = []
	# y_axis_min = []
	# # y_axis = ts.period_data
	# for i in range(0, len(ts.period_data)):
	# # 	x_axis.append(i)
	# for i in range(0, 96):
	# 	x_axis.append(i)
	# 	y_axis_min.append(ts.min_data[i])
	# 	y_axis.append(ts.period_data[i])
	# plt.plot(x_axis, y_axis, c='r')
	# plt.plot(x_axis, y_axis_min, c='g')
	# plt.plot(label,c='r',label='data')
	# plt.plot(fortrain,c='g',label='svr model')

	#展示结果
	# plt.plot(y_test,c='r')
	# plt.plot(y_hat,c='g')

	# str_figname = Figure_PATH + 'Fig_' + mod_name + '.png'
	# plt.savefig(str_figname)

	# # plt.ylim(0, 1000)
	# # ytick_1 = list(np.arange(0, 1000, 100))

	# # plt.ylabel('y')
	# # print(ytick_1)
	# # plt.yticks(ytick_1)

	# plt.show()
	# for i in range(0, 1440):
	# 	x_axis.append(i)
	# 	# print(ts_map.day[0][i])
	# 	# y_axis.append(ts_map.day[0][i*60])
	# for st in ts_map.stations:
	# 	y_axis = []
	# 	for i in range(0, 1440):
	# 		y_axis.append(ts_map.day[st[2]][i*60])
	# 	plt.plot(x_axis, y_axis)
	# 	plt.title("Station: "+str(ts_map.stations[0])+","+str(ts_map.stations[1])) # 图形标题
	# 	plt.xlabel("Minutes of a day.") # x轴名称
	# 	plt.ylabel("Load of the station.") # y 轴名称
	# 	str_figname = Figure_PATH + 'Fig_' + str(st[2]) + '.png'
	# 	plt.savefig(str_figname)
	# 	plt.clf()
	print("plot complete")
