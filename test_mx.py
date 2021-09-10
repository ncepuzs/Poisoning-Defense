import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dsets

import numpy as np
import logging

import time 
import 

import torch.nn as nn
import torch.optim as optim

def generate_poison(classifier, inv_model, param_path, train_batch, poisoned_rate, nz, device, k_wargs)
	# load model 
	# inversion = nn.DataParallel(Inversion(nz=args.nz, truncation=args.truncation, c=args.c, h=h_value_list)).to(device)
	optimizer = optim.Adam(inversion.parameters(), lr=0.001, betas=(0.5, 0.999), amsgrad=True)

	inversion_path = param_path
	checkpoint = torch.load(inversion_path)
	inversion.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    best_cl_acc = checkpoint['best_recon_loss']
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded inversion checkpoint '{}' (epoch {}, loss {:.4f})".format(path, epoch, best_recon_loss))


    data_batch = train_batch

	#========================initial poisoning example=============================
	round_num = 20 # number of round to generate a poisoning example
	group_size = 100 # number of normal data used to compute the loss of inversion model	classifier.eval()

	img_0 = data_batch[0:1]
	vector_p = classifier(img_0, release=Ture) # poisoned vector
	img_p = data_batch[1:2].clone() # poisoned label (the label of next img)

	normal_data = data_batch[0:group_size]

	# ===============update inversion using poisoning example======================

	inversion.train()
	optimizer.zero_grad()
	reconstruction = inversion(vector_p)
	loss = F.mse_loss(reconstruction, img_i, reduction='sum').item()
	loss.backward()
	optimizer.step()
	print("The first update of inversion model using poisoning example has been completed")


	#========================compute loss on normal data========================
	classifier.eval()
	inversion.eval()
	pre_loss = 0
	normal_data = normal_data.to(divice)
	normal_vector = classifier(normal_data) 

	reconstruction = inversion(normal_vector)
	pre_loss = F.mse_loss(reconstruction, normal_data, reduction='sum').item()


	pre_loss /= group_size * 32 * 32
	print('Before poisoning: Group average loss is \t{}').format(pre_loss)
	logging.info('Before poisoning: Group average loss is \t{}').format(pre_loss)


	# generate poisoning example from data_p
	for round_i in range(round_num):

		# compute poisoning sample gradient
		dir = torch.zeros(vector_p.shape)
		for grad_round in range(len(vector_p)):
			vector_p_tmp = vector_p.clone()
			vector_p_tmp[grad_round] += 0.001

			inversion.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			inversion.train()
			# optimizer = optim.Adam(inversion.parameters(), lr=0.001, betas=(0.5, 0.999), amsgrad=True)
			optimizer.zero_grad()

			reconstruction = inversion(vector_p)
			loss = F.mse_loss(reconstruction, img_p)

			loss.backward()
			optimizer.step()

			# ivnersion update complete, and then start compute the loss on normal data
			loss = 0
			normal_data = normal_data.to(divice)
			normal_vector = classifier(normal_data) 

			reconstruction = inversion(normal_vector)
			loss = F.mse_loss(reconstruction, normal_data, reduction='sum').item()
		dir[grad_round] = np.sign(loss-pre_loss)

		# update poisoned data
		tmp = vector_p + dir*0.2
		tmp[tmp > 1] = 1
	    tmp[tmp < 0] = 
	    vector_p = tmp

	    # first round of generation of poisoning sample complete.
	    # update the inversion model using poisoning example
	    inversion.load_state_dict(checkpoint['model'])
	    optimizer.load_state_dict(checkpoint['optimizer'])
	    inversion.train()
	    optimizer.zero_grad()
	    reconstruction = inversion(vector_p)
		loss = F.mse_loss(reconstruction, img_p)

		loss.backward()
		optimizer.step()


	    # compute the pre_loss of normal data in updated inversion
	    inversion.eval()
		loss = 0
		normal_data = normal_data.to(divice)
		normal_vector = classifier(normal_data) 

		reconstruction = inversion(normal_vector)
		loss = F.mse_loss(reconstruction, normal_data, reduction='sum').item()
		pre_loss = loss

	return vector_p



		





