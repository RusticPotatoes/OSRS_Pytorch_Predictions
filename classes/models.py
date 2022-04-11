import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		#  stateless LSTM
		# self.hidden_cell = (torch.zeros(1,1,self.hidden_dim),
		# 					torch.zeros(1,1,self.hidden_dim))
		
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):  
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

		# detatch=(h0.detach(), c0.detach())
		out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
		out = self.fc(out[:, -1, :]) 
		return out

	def forecast(self, dataset, lookback, fut_pred):
		self.eval()
		torch_tensor_dataset = torch.from_numpy(dataset).type(torch.Tensor)
		for i in range(fut_pred):
			seq = torch_tensor_dataset[-1:,-lookback:,:]
			with torch.no_grad():
				self.hidden = (
					torch.zeros(1, 1, self.hidden_dim),
					torch.zeros(1, 1, self.hidden_dim)
					)
				result = self(seq)  # .item()#tuple?
				torch_tensor_dataset = torch.cat(
					(torch_tensor_dataset, ((result)[np.newaxis,:])),
					dim=1
					)
		return (torch_tensor_dataset[-1:,-fut_pred:,:]).detach().numpy()


class GRU(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
		super(GRU, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		
		self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
		out, (hn) = self.gru(x, (h0.detach()))
		out = self.fc(out[:, -1, :]) 
		return out

	def forecast(self, dataset, lookback, fut_pred):
		self.eval()
		torch_tensor_dataset = torch.from_numpy(dataset).type(torch.Tensor)
		for i in range(fut_pred):
			seq = torch_tensor_dataset[-1:,-lookback:,:]
			with torch.no_grad():
				self.hidden = (
					torch.zeros(1, 1, self.hidden_dim),
					torch.zeros(1, 1, self.hidden_dim)
					)
				result = self(seq)#.item()#tuple?
				torch_tensor_dataset = torch.cat(
					(torch_tensor_dataset, ((result)[np.newaxis,:])),
					dim=1
					)
		return (torch_tensor_dataset[-1:,-lookback:,:]).detach().numpy()

