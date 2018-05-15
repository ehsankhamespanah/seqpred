import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from model import *
import ipdb

from load_data import *

EPOCHS = 50
BATCH_SIZE = 16
PRINT_EVERY = 100
CUDA = False
criterion = nn.CrossEntropyLoss()


def test_file(fname, modelfile):
	model = MultiAttrEncoder(9,9,256, CUDA)	
	model.load_state_dict(torch.load(modelfile))
	ds = EventsDataset(fname)
	sampler = torch.utils.data.sampler.SequentialSampler(ds)
	trainloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, sampler = sampler)
	rc = []
	for i, data in enumerate(trainloader):
		data = Variable(data)
		if CUDA:
			data = data.cuda()
		logits = model(data)
		rc.append(logits)

	return rc



def main():
	ds = EventsDataset('events-array-1.txt')
	sampler = torch.utils.data.sampler.RandomSampler(ds)
	trainloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, sampler = sampler)
	if CUDA:
		model = model.cuda()
	optimizer = optim.SGD(model.parameters(), lr=0.0006, momentum=0.9)

	for epoch in range(EPOCHS):
		for i, data in enumerate(trainloader):
			data = Variable(data)
			if CUDA:
				data = data.cuda()
			optimizer.zero_grad()
			logits = model(data)		
			aux_logits = logits[:,:-1,:,:].contiguous().view(-1,2)
			aux_labels = data[:,1:,:].contiguous().view(-1)
			loss = criterion(aux_logits,aux_labels)		
			loss.backward()
			optimizer.step()
			if i > 0 and not i % PRINT_EVERY:
				print('[{}] Loss: {}'.format(i,loss))
if __name__=='__main__':
	main()
