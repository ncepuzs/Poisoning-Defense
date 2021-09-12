import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dsets
import torch.nn.functional as F

import numpy as np
import logging

import time
import math

import torch.nn as nn
import torch.optim as optim


def generate_poison(classifier, inversion, checkpoint, train_batch, device, group_size=10):
	classifier.eval()
	# load model
	# inversion = nn.DataParallel(Inversion(nz=args.nz, truncation=args.truncation, c=args.c, h=h_value_list)).to(device)
	optimizer2 = optim.Adam(inversion.parameters(), lr=0.001, betas=(0.5, 0.999), amsgrad=True)

	inversion.load_state_dict(checkpoint['model'])
	optimizer2.load_state_dict(checkpoint['optimizer'])

	data_batch = train_batch

	# ========================initial poisoning example=============================
	round_num = 100  # number of round to generate a poisoning example
	group_size = group_size  # number of normal data used to compute the loss of inversion model	classifier.eval()

	img_0 = data_batch[0:1]
	classifier.eval()
	with torch.no_grad():
		vector_p = classifier(img_0, release=True) # poisoned vector
	img_p = data_batch[1:2].clone().to(device)  # poisoned label (the label of next img)

	normal_data = data_batch[0:group_size]
	# print("type and shape of img_0:{},{}".format(type(img_0), img_0.shape))
	# print("type and shape of normal_data:{},{}".format(type(normal_data), normal_data.shape))

	# ===============update inversion using poisoning example======================

	inversion.train()
	optimizer2.zero_grad()
	reconstruction = inversion(vector_p).to(device)
	loss = F.mse_loss(reconstruction, img_p, reduction='sum')
	#print(loss)
	loss.backward()
	#print(loss)
	#print("type of loss:\t{}, shape:\t{}".format(type(loss), loss.shape))
	optimizer2.step()
	# print("=====The first update of inversion model using poisoning example has been completed=====")

	# ========================compute loss on normal data========================
	inversion.eval()
	with torch.no_grad():
		pre_loss = 0
		normal_data = normal_data.to(device)
		normal_vector = classifier(normal_data, release=True)
		#print("type of normal_data:\t{}, shape:\t{}".format(type(normal_data), normal_data.shape))
		#print("type of normal_vector:\t{}, shape:\t{}".format(type(normal_vector), normal_vector.shape))

		reconstruction = inversion(normal_vector)
		pre_loss = F.mse_loss(reconstruction, normal_data, reduction='sum').item()
		#print("type of pre_loss:{}, shape:{}".format(type(pre_loss), pre_loss.shape))
		#print(pre_loss)
		#print("loss:", loss)

		pre_loss /= group_size * 32 * 32
		# print('Before poisoning: Group average loss is \t{}'.format(pre_loss))
		logging.info('Before poisoning: Group average loss is \t{}'.format(pre_loss))

	# generate poisoning example from data_p
	for round_i in range(round_num):
		# compute poisoning sample gradient
		dir = torch.zeros(vector_p.shape)

		for grad_round in range(len(vector_p[0])):
			vector_p_tmp = vector_p.clone()
			vector_p_tmp[0][grad_round] += 0.00001

			inversion.load_state_dict(checkpoint['model'])
			optimizer2.load_state_dict(checkpoint['optimizer'])
			inversion.train()
			# optimizer = optim.Adam(inversion.parameters(), lr=0.001, betas=(0.5, 0.999), amsgrad=True)
			optimizer2.zero_grad()

			reconstruction = inversion(vector_p_tmp)
			loss_1 = F.mse_loss(reconstruction, img_p)

			loss_1.backward()
			optimizer2.step()

			# ivnersion update complete, and then start compute the loss on normal data
			loss = 0
			# normal_data = normal_data.to(device)
			# normal_vector = classifier(normal_data, release=True)

			reconstruction = inversion(normal_vector)
			loss = F.mse_loss(reconstruction, normal_data, reduction='sum').item()
			dir[0][grad_round] = np.sign(loss - pre_loss)

		# update poisoned data
		if round_num <= 80:
			tmp = vector_p + (dir * 0.005).to(device)
		else:
			tmp = vector_p + (dir * 0.001).to(device)
		tmp[tmp > 1] = 1
		tmp[tmp < 0] = 0
		vector_p = F.softmax(tmp, dim=1)

		# print("vector_p:,", vector_p)
		# first round of generation of poisoning sample complete.
		# update the inversion model using poisoning example
		inversion.load_state_dict(checkpoint['model'])
		optimizer2.load_state_dict(checkpoint['optimizer'])
		inversion.train()
		optimizer2.zero_grad()
		reconstruction = inversion(vector_p)
		loss = F.mse_loss(reconstruction, img_p)
		# print(loss)

		loss.backward()
		optimizer2.step()

		# compute the pre_loss of normal data in updated inversion
		inversion.eval()
		with torch.no_grad():
			loss = 0
			# normal_data = normal_data.to(device)
			# normal_vector = classifier(normal_data, release=True)

			reconstruction = inversion(normal_vector)
			loss = F.mse_loss(reconstruction, normal_data, reduction='sum').item()
			pre_loss = loss/ (group_size * 32 *32)

	return vector_p, normal_vector
