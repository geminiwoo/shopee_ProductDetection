import os
import csv
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from argparse import ArgumentParser
from Model import Model

def str2bool(value):
    if isinstance(value, bool):
       return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_args():
	# you can add any args as you need here
	parser = ArgumentParser()
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--num_epochs', default=20, type=int)
	parser.add_argument('--save_path', default='./model.pth')
	parser.add_argument('--is_train', type=str2bool, nargs='?', const=True, default=True)
	return parser.parse_args()

def getDataLoader(args, transform, split_ratio=0.8):
    if args.is_train:
        dataset = torchvision.datasets.ImageFolder(root='train/train/', transform=transform)
        train_size = int(split_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
        return train_loader, valid_loader
    else:
        dataset = torchvision.datasets.ImageFolder(root='test/', transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
        return test_loader, dataset

def writeCSV(filenames, predictions):
	table = list(zip(filenames, predictions))
	with open('output.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['filename', 'category'])
		writer.writerows(table)

def main():
	args = init_args()
	# image preprosessing
	transform = transforms.Compose([transforms.Resize((64, 64)), 
                                	transforms.ToTensor(), 
                                	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	if args.is_train:
		train_loader, valid_loader = getDataLoader(args, transform)
		model = Model(args)
		model.train(train_loader, valid_loader)
	else:
		test_loader, test_dataset = getDataLoader(args, transform)
		filenames = [filename.replace('test/test','') for filename, _ in test_dataset.imgs]
		model = Model(args)
		model.load(args.save_path)
		predictions = model.predict(test_loader)
		writeCSV(filenames, predictions)
		print('finish writing the CSV!')

if __name__ == '__main__':
	main()