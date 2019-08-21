import torch as t
from torch.nn import functional as F
from torch.autograd import Variable
from tools.ListToTensor import ListToTensor

def train_once(data_loader_train, net, optimizer, cost):
	run_loss = 0.0
	run_correct = 0.0

	for data in iter(data_loader_train):
		X_train, X_label = data
		label = []
		for la in X_label:
			label.append(la)
		label = ListToTensor(label)

		X_train = Variable(X_train)
		X_label = Variable(label)
		optimizer.zero_grad()
		outputs = net(X_train)

		_, pred = t.max(F.softmax(outputs, dim=1).data, 1)
		loss = cost(outputs, X_label)
		run_loss += loss.data
		run_loss = run_loss.item()

		loss.backward()
		optimizer.step()
		run_correct += (pred == X_label.data).sum()
		corr = (1.*run_correct).item()
	return run_loss, corr

