import numpy as np
from torch.utils.data import Dataset


class EventsDataset(Dataset):
	def __init__(self, fname, seq_length=100):
		self.all_data = np.genfromtxt(fname).astype(np.long)
		self.seq_length = seq_length
	
	def __len__(self):
		return len(self.all_data) - (self.seq_length-1)

	def __getitem__(self, idx):
		return self.all_data[idx:idx+self.seq_length,:]
