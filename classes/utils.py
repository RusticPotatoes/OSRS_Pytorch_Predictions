
import os
import imageio
import itertools
import pandas as pd
import shutil
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler,Normalizer
import torch
import torch.nn as nn
import sys


# ================== Utility FUNCTIONS ================== 
# Unnormalizing the data (so we can see actual prices in GP)
def getIDFromName(df, name):
	return (df[df['name'] == name].item_id.item())


def getNameFromID(df, id):
	return (df[df['item_id'] == id].name.item())


def gif_from_png_dir(item_to_predict, img_dir):
	images = []
	timestr = time.time()#('%Y%m%d-%H%M%S')
	for file_name in sorted(os.listdir(img_dir)):
		if file_name.endswith('.png'):
			file_path = os.path.join(img_dir, file_name)
			images.append(imageio.imread(file_path))
	gifpath = os.path.join(img_dir, '{}_{}.gif'.format(item_to_predict,timestr))
	imageio.mimsave(gifpath, images, fps=1)


def clear_pngs(img_dir):
	if not os.path.exists(img_dir):
		return
	for file in os.listdir(img_dir):
		if file.endswith('.png'):
			os.remove(os.path.join(img_dir, file)) 


def save_plot_to_png(input_plot, filename, folderpath):   			
	mode = 0o777
	global img_dir
	if folderpath is not None: #check subdir path and make it, append the subdir to img_dir
		if not os.path.exists(folderpath): os.makedirs(folderpath, mode)
	# print(os.path.join(folderpath, filename))
	input_plot.savefig(os.path.join(folderpath, filename.replace(" ", "_")))  # '{}_{}.png'.format(item_to_predict,index))


def clear_folder(folder):
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))


def split_data_2d(dataset, lookback):
	data_raw = dataset.to_numpy()  # convert to numpy array
	data = []
	
	# create all possible sequences of length seq_len
	for index in range(len(data_raw) - lookback): 
		data.append(data_raw[index: index + lookback])
	
	data = np.array(data);
	test_set_size = int(np.round(0.2*data.shape[0]));
	train_set_size = data.shape[0] - (test_set_size);
	
	x_train = data[:train_set_size, :-1]
	y_train = data[:train_set_size, -1]
	
	x_test = data[train_set_size:, :-1]
	y_test = data[train_set_size:, -1]
	
	return [x_train, y_train, x_test, y_test]


def split_data_3d(dataset, lookback, lookforward=0):
	if not isinstance(dataset, np.ndarray):
		data_raw = dataset.to_numpy()  # convert to numpy array
	else:
		data_raw = dataset
	data = []
	
	# create all possible sequences of length seq_len
	for index in range(len(data_raw) - lookback): 
		data.append(data_raw[index: index + lookback])
	
	data = np.array(data);
	test_set_size = int(np.round(0.2*data.shape[0]));
	train_set_size = data.shape[0] - (test_set_size);
	# print(f":{train_set_size}:-1,:")

	x_train = data[:train_set_size, :-1, :]
	y_train = data[:train_set_size, -1, :]
	
	x_test = data[train_set_size:, :-1, :]
	y_test = data[train_set_size:, -1, :]

	if lookforward > 0:
		forcast = 0
	
	return x_train, y_train, x_test, y_test


def split_data_3d_testonly(dataset, lookback, lookforward=0):
	if not isinstance(dataset, np.ndarray):
		data_raw = dataset.to_numpy()  # convert to numpy array
	else:
		data_raw = dataset
	data = []
	
	# create all possible sequences of length seq_len
	for index in range(len(data_raw) - lookback): 
		data.append(data_raw[index: index + lookback])
	
	data = np.array(data);
	test_set_size = int(np.round(0.2*data.shape[0]));
	train_set_size = data.shape[0] - (test_set_size);
	# print(f":{train_set_size}:-1,:")
	
	x_test = data[:, :-1, :]
	y_test = data[:, -1, :]

	if lookforward > 0:
		forcast = 0
	
	return x_test, y_test

def scale_data(scaler, transformer, data, inverse=False):
	if isinstance(data, pd.DataFrame):	
		columns = data.columns
		index = data.index
	#print(type(data))
	if scaler is None:
		scaler = MinMaxScaler((-1,1))
		#MinMaxScaler(feature_range=(-1,1))
	if transformer is None:
		transformer =  Normalizer().fit(data)#scaler.fit_transform(data)
	if inverse:
		data = scaler.inverse_transform(data)
	else:
		data = scaler.transform(data)
		# data = transformer.transform(data)
	if isinstance(data, pd.DataFrame):	
		data = pd.DataFrame(data, columns=columns, index=index)
	return scaler, transformer, data


def forecast(model,dataset,lookback,fut_pred): # abstracted to models 
	model.eval()
	torch_tensor_dataset = torch.from_numpy(dataset).type(torch.Tensor)
	for i in range(fut_pred):
		seq = torch_tensor_dataset[-1:,-lookback:,:]
		#seq_2d = seq[0,:,:]
		# print(seq.shape)
		with torch.no_grad():
			model.hidden = (torch.zeros(1, 1, model.hidden_dim),
							torch.zeros(1, 1, model.hidden_dim))
			result= model(seq)#.item()#tuple?
			#print(results)
			#print(results.shape)
			torch_tensor_dataset = torch.cat(
					(torch_tensor_dataset,(results)[np.newaxis,:]),
					dim=1
				)
	return (torch_tensor_dataset).detach().numpy()


def normalizer(df):
	df_std = df.std()
	df_mean = df.mean()
	normalized_df=(df-df_mean)/df_std
	return normalized_df, df_std, df_mean


def unnormalizer(df, df_std, df_mean):
	#need to make df 2d instead of 3d...
	std=df_std[0]
	mean=df_mean[0]
	unnormalized_df=(df*std)+mean
	return unnormalized_df

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
