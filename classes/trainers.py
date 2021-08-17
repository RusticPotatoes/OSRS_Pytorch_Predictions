import time 
import numpy as np

def train(model, criterion, optimiser, num_epochs, x_train, y_train, verbose=False):
	hist = np.zeros(num_epochs)
	start_time = time.time()

	for t in range(num_epochs):
		y_train_pred = model(x_train)

		loss = criterion(y_train_pred, y_train)
		if verbose: 
			if t%25==1:
		  		print("Epoch ", t, "MSE: ", loss.item())
		hist[t] = loss.item()

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
	print("Epoch ", t, "MSE: ", loss.item())
	training_time = time.time()-start_time
	if verbose: print("Training time: {}".format(training_time))
	return y_train_pred, hist