from CNN import CNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Model():
	def __init__(self, args):
		self.args = args
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model = CNN().to(self.device)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)

	def train(self, train_loader, valid_loader):
		min_accuracy = -np.inf
		for epoch in range(self.args.num_epochs):
			running_loss = 0.0
            
			print('Run Training...')
			for step, data in enumerate(train_loader, 0):
				images, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
				self.optimizer.zero_grad()

                # forward + backward + optimize
				outputs = self.model(images)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

                # print statistics
				running_loss += loss.item()
				if step % 200 == 199:    # print every 200 mini-batches
					print('[epoch: %d, steps: %5d] loss: %.3f' %(epoch + 1, step + 1, running_loss / 200))
					running_loss = 0.0
              
			print('Run Validation...')
			total = 0
			correct = 0
			self.model.eval()
			with torch.no_grad():
				for data in valid_loader:
					images, labels = data[0].to(self.device), data[1].to(self.device)
					outputs = self.model(images)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
			valid_acc = correct / total
			print('Accuracy of the network on the valid images: %d %%' % (100 * valid_acc))
            
			if valid_acc > min_accuracy:
				print('save the CNN to ', self.args.save_path)
				torch.save(self.model.state_dict(), self.args.save_path)
				min_accuracy = valid_acc
            
		print('Finished Training!')

	def load(self, save_path):
		print('load the model from', save_path)
		self.model.load_state_dict(torch.load(save_path))

	def predict(self, test_loader):
		predictions = []
		with torch.no_grad():
			for images, _ in test_loader:
				outputs = self.model(images.to(self.device))
				_, prediction = torch.max(outputs, 1)
				predictions = predictions + prediction.tolist()
		return predictions