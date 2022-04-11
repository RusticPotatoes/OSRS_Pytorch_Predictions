#https://www.kaggle.com/rodsaldanha/stock-prediction-pytorch
from classes import utils, plot, trainers, models
from classes.wrapper import PricesAPI
from classes.models import LSTM
import numpy as np 
import pandas as pd 
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import math, time
from sklearn.metrics import mean_squared_error


# known project folders 
parent_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(parent_dir,"models")
features_dir = os.path.join(models_dir,"features")
img_dir = os.path.join(parent_dir,'imgs')
forecast_dir=os.path.join(img_dir,'forecasts')
training_dir=os.path.join(img_dir,'trainings')
prices_dir=os.path.join(img_dir,'prices')
data_dir = os.path.join(parent_dir,'data')
items_to_predict_file=os.path.join(parent_dir,"items_to_predict.csv")

#low_price = buy_average = avgLowPrice
#high_price = sell_average = avgHighPrice
#low_volume= buy_quantity = lowPriceVolume
#high_volume = sell_quantity = highPriceVolume

def main():
	global parent_dir,models_dir,features_dir,img_dir,data_dir,items_to_predict_file,forecast_dir,training_dir,prices_dir
	
	##########################
	#### config variables ####
	##########################
	save_img=True #saves to img folder
	lookback=40 #choose a sequence length
	num_epochs = 100
	verbose = True
	fut_pred = 20
	##########################
	##########################

	##########################
	###### DO NOT TOUCH ######
	##########################
	input_dim = 1
	hidden_dim = 32
	num_layers = 2
	output_dim = 1
	##########################
	##########################
		
	# import the items to use 
	items_to_predict_df = pd.read_csv(items_to_predict_file)
	items_to_predict_df_names = items_to_predict_df['name']
	features_to_predict_df_names=items_to_predict_df.columns.values[1:3]

	# query runelite for prices 
	apimapping = PricesAPI("GEPrediction-OSRS","GEPRediction-OSRS")
	item_mapping_df = apimapping.mapping_df()

	count = 0
	for item_to_predict in items_to_predict_df_names:
		for feature_to_predict_name in features_to_predict_df_names:
			# for each item
			apitimeseries = PricesAPI("OSRS_PYTORCH_PREDICTIONS","OSRS_PYTORCH_PREDICTIONS")
			runelite_timeseries_df = apitimeseries.timeseries_df("5m", utils.getIDFromName(item_mapping_df,item_to_predict)) 
			# add the name column back
			runelite_timeseries_df['name'] = item_to_predict
			# epoch unix to datetime
			runelite_timeseries_df['timestamp']=pd.to_datetime(runelite_timeseries_df['timestamp'], unit='s')
			# fix 0's to ba nans 
			runelite_timeseries_df.replace({'0':np.nan, 0:np.nan}, inplace=True)

			# set timestamp as index
			runelite_timeseries_df.set_index(['timestamp'], inplace=True)
			price_all = pd.DataFrame(runelite_timeseries_df, columns=features_to_predict_df_names)
			price = price_all[[feature_to_predict_name]]

			if verbose: price.info()

			# clear NaNs
			price=price.dropna()
	
			if save_img:
				#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
				item_to_predict_dataset_plot = plot.plot_single(dataset= price[feature_to_predict_name].values, 
								xaxisticks=(range(0,price.shape[0],30)), # stopped here price.shape is wrong, from 0 to shape range (300) increment 30
								xaxisdata=((price.index.values)[::30]), #loc[::(price.shape[0])]
								title=(item_to_predict), 
								xlabel='Date', 
								ylabel=f'{items_to_predict_df.columns[1]}')
				utils.save_plot_to_png(item_to_predict_dataset_plot, f"{item_to_predict}_{feature_to_predict_name}_history.png",prices_dir)

			# normalizing 
			# should not use min max scaling for price data that has no theoretical max value
			## 			scaler = MinMaxScaler(feature_range=(-1,1))
			##			price_reshaped= price.values.reshape(-1,1)
			# 			#scale the data
			#			price_scaled =scaler.fit_transform(price_reshaped)
			# standardization by normalization eg (values-mean)/std
  			# normalized_df, df_std, df_mean
			price_normalized_df, price_std, price_mean = utils.normalizer(price)
			price_reshaped= price.values.reshape(-1,1)
			price_normalized_reshaped_df= price_normalized_df.values.reshape(-1,1)
			price_scaled=price_normalized_reshaped_df

			# needed for later forecasting 1,X,1 shape eg 1,300,1
			price_scaled_reshaped_3d = price_scaled.reshape(1,price.shape[0],1)

			#split the data up 
			x_train, y_train, x_test, y_test = utils.split_data_3d(price_scaled,lookback)

			if verbose: print('x_train.shape = ',x_train.shape)
			if verbose: print('y_train.shape = ',y_train.shape)
			if verbose: print('x_test.shape = ',x_test.shape)
			if verbose: print('y_test.shape = ',y_test.shape)

			# setup torch tensors
			x_train_tensor = torch.from_numpy(x_train).type(torch.Tensor)
			#if verbose: print(type(x_train_tensor))
			x_test_tensor = torch.from_numpy(x_test).type(torch.Tensor)
			#if verbose: print(type(x_test_tensor))

			# lstm y train and test
			y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
			y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
			# list for results
			lstm_results=[]
			gru_results = []
			# gru vars
			y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
			y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

			################# use the LSTM model #################
			model_lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
			
			criterion = torch.nn.MSELoss(reduction='mean')
			optimiser_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.01)
			hist_lstm = np.zeros(num_epochs)

			# train predictions
			y_train_pred_lstm, hist_lstm = trainers.train(model_lstm, criterion, optimiser_lstm, num_epochs, x_train_tensor, y_train_lstm, verbose)
			# prediction lstm and original rrom lstm 
			# inverse predictions from train
			#y_train_pred_lstm_inverse = scaler.inverse_transform(y_train_pred_lstm.detach().numpy())
			y_train_pred_lstm_inverse = utils.unnormalizer(y_train_pred_lstm.detach().numpy(),price_std,price_mean)
			y_train_pred_lstm_inverse_df = pd.DataFrame(y_train_pred_lstm_inverse)


			#y_train_lstm_inverse = scaler.inverse_transform(y_train_lstm.detach().numpy())
			y_train_lstm_inverse = utils.unnormalizer(y_train_lstm.detach().numpy(),price_std,price_mean)
			y_train_lstm_inverse_df = pd.DataFrame(y_train_lstm_inverse)
			# plot original and prediction graphs with training loss
			if save_img:
				dual_plot_lstm = plot.plot_dual(original=y_train_lstm_inverse_df, predict=y_train_pred_lstm_inverse_df, hist=hist_lstm, modelname="LSTM", title=f"{item_to_predict}", xlabel="Date", ylabel=f"Gold (GP) [{feature_to_predict_name}]")
				utils.save_plot_to_png(dual_plot_lstm, f"lstm_dual_{item_to_predict}_{feature_to_predict_name}.png",training_dir)

			#test and then inverse test predictions 
			model_lstm.eval()
			y_test_pred_lstm= model_lstm(x_test_tensor) 
			##########################################
			#y_test_pred_lstm_inverse = scaler.inverse_transform(y_test_pred_lstm.detach().numpy())
			y_test_pred_lstm_inverse = utils.unnormalizer(y_test_pred_lstm.detach().numpy(),price_std,price_mean)
			y_test_pred_lstm_inverse_df=pd.DataFrame(y_test_pred_lstm_inverse)

			#y_test_lstm_inverse = scaler.inverse_transform(y_test_lstm.detach().numpy())
			y_test_lstm_inverse = utils.unnormalizer(y_test_lstm.detach().numpy(),price_std,price_mean)
			y_test_lstm_inverse_df = pd.DataFrame(y_test_lstm_inverse)

			# calculate root mean squared error
			trainScore_lstm = math.sqrt(mean_squared_error(y_train_lstm_inverse[:,0], y_train_pred_lstm_inverse[:,0]))
			if verbose: print('Train Score: %.2f RMSE' % (trainScore_lstm))
			testScore_lstm = math.sqrt(mean_squared_error(y_test_lstm_inverse[:,0], y_test_pred_lstm_inverse[:,0]))
			if verbose: print('Test Score: %.2f RMSE' % (testScore_lstm))
	
			lstm_results.append([trainScore_lstm,testScore_lstm])
			########################################################################################################

			#future prediction https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
			
			#use learned model, use forecast to predict into future, forecast does not train model
			z_forecast_lstm_scaled_reshaped_3d = model_lstm.forecast(dataset=price_scaled_reshaped_3d,lookback=lookback,fut_pred=fut_pred)#model_lstm.forecast(dataset=z_future_tensor,lookback=lookback,fut_pred=fut_pred)
			#remove scale by running inverse transform, and reshape back to 2d array
			#z_forecast_lstm_inverse = scaler.inverse_transform(z_forecast_lstm_scaled_reshaped_3d[0,-fut_pred:,:])
			z_forecast_subset=z_forecast_lstm_scaled_reshaped_3d[0,-fut_pred:,:]
			z_forecast_lstm_inverse = utils.unnormalizer(z_forecast_lstm_scaled_reshaped_3d[0,-fut_pred:,:],price_std,price_mean)
			z_test_pred_lstm_inverse_df=pd.DataFrame(z_forecast_lstm_inverse)
			#if verbose: print(z_test_pred_lstm_inverse_df.tail(fut_pred+1))

			##########################################################################################################################

			# shift train predictions for plotting
			trainPredictPlot = np.empty((price.shape[0]+fut_pred,1))
			trainPredictPlot[:, :] = np.nan
			print(f"trainPredictPlot: {lookback}: {len(y_train_pred_lstm_inverse)+lookback} , :")
			trainPredictPlot[lookback:len(y_train_pred_lstm_inverse)+lookback, :] = y_train_pred_lstm_inverse

			# shift test predictions for plotting
			testPredictPlot =  np.empty((price.shape[0]+fut_pred,1))
			testPredictPlot[:, :] = np.nan
			print(f"test predict plot: {len(y_train_pred_lstm_inverse)+lookback} : {len(price)} , :")
			testPredictPlot[len(y_train_pred_lstm_inverse)+lookback:len(price), :] = y_test_pred_lstm_inverse
			
			forecastPredictPlot=  np.empty((price.shape[0]+fut_pred,1))
			forecastPredictPlot[:, :] = np.nan
			print(f"forecast predict plot: {price.shape[0]} : {price.shape[0]+fut_pred} , :")
			forecastPredictPlot[price.shape[0]:price.shape[0]+fut_pred,:]=z_forecast_lstm_inverse

			originalPlot =  np.empty((price.shape[0]+fut_pred,1))
			originalPlot[:, :] = np.nan
			print(f"{len(price_reshaped)}:,:")
			originalPlot[0:len(price_reshaped),:]= price_reshaped

			predictions = trainPredictPlot
			predictions = np.append(predictions, testPredictPlot, axis=1)
			predictions = np.append(predictions, forecastPredictPlot, axis=1)
			predictions = np.append(predictions, originalPlot, axis=1)
			result = pd.DataFrame(predictions)

			test_pred_fig = plot.plot_multi(result,title=f"{item_to_predict} (LSTM)",feature=feature_to_predict_name)
			if not os.path.exists(forecast_dir):
				mode = 0o777
				os.makedirs(forecast_dir)
			output_img=os.path.join(forecast_dir,f"lstm_dual_{item_to_predict}_{feature_to_predict_name}_train_test_forcast.png").replace(" ","_")
			print(output_img)
			test_pred_fig.write_image(engine="kaleido", file=output_img)

		######################################################

		############## TODO: Use the GRU model ###############

		######################################################

		count+=1
	return

if __name__ == "__main__":
	main()
