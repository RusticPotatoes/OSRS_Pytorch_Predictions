import time 
import numpy as np
import torch
import math

def train(model, criterion, optimiser, num_epochs, x_train, y_train, verbose=False):
	hist = np.zeros(num_epochs)
	start_time = time.time()
	if torch.cuda.is_available():
		model= model.cuda()
		x_train=x_train.cuda()
		y_train=y_train.cuda()
		print("CUDA Enabled!")

	for t in range(num_epochs):
		y_train_pred = model(x_train)
		loss = criterion(y_train_pred, y_train)
		if verbose: 
			if t%(math.floor(num_epochs/10))==1:
				print(f"Epoch {t} MSE: {loss.item()}")
		hist[t] = loss.item()
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
	print(f"Epoch {t} MSE: {loss.item()}")
	training_time = time.time()-start_time
	if verbose: print("Training time: {}".format(training_time))
	if torch.cuda.is_available():
		model= model.cpu()
		x_train=x_train.cpu()
		y_train=y_train.cpu()
		y_train_pred=y_train_pred.cpu()
	return y_train_pred, hist