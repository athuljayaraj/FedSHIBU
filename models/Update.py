#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from collections import UserDict
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
from utils.similarity import cka, gram_linear
from utils.train_utils import get_model

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False, validation_dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def shibu_aggregate(self, set_of_w_local):
        print('Doing SHIBU aggregation')
        local_model = get_model(self.args)
        client_representations_tensor = []
        for w_local in set_of_w_local:
            local_model.load_state_dict(w_local)
            representations = torch.empty(0, 64).to(self.args.device) # TODO remove hardcoded argument in torch.empty
            for idx, (data, target) in enumerate(self.ldr_train):
                representation = local_model.extract_features(data.to(self.args.device))
                representations = torch.cat([representations, representation], dim=0)
                break # TODO fix CUDA memory leak and remove break
            client_representations_tensor.append(representations)
        client_representations_tensor = torch.stack(client_representations_tensor, dim=0).to(self.args.device)

        print(client_representations_tensor.shape)

        similarity_matrix = torch.zeros(self.args.num_users, self.args.num_users)

        for idx1 in range(len(client_representations_tensor)):
            for idx2 in range(idx1, len(client_representations_tensor)):
                similarity_matrix[idx1][idx2] = cka(gram_x=gram_linear(client_representations_tensor[idx1]), gram_y=gram_linear(client_representations_tensor[idx2]))
                similarity_matrix[idx2][idx1] = similarity_matrix[idx1][idx2]

        normalized_similarity_matrix = self.normalize(similarity_matrix)
        client_wise_updated_weights = self.weighted_average(set_of_w_local, normalized_similarity_matrix)
        
        return client_wise_updated_weights

    def normalize(self, similarity_matrix):
        normalized_similarity_matrix = torch.clone(similarity_matrix)
        for user_idx in range(self.args.num_users):
            amin, amax = min(similarity_matrix[user_idx]), max(similarity_matrix[user_idx])
            for i, val in enumerate(similarity_matrix[user_idx]):
                normalized_similarity_matrix[user_idx][i] = (val - amin) / (amax - amin)
        return normalized_similarity_matrix

    def weighted_average(self, set_of_w_local, similarity_matrix):
        client_wise_updated_weights = {}
        for user_idx in range(self.args.num_users):
            updated_w_glob = get_model(self.args).state_dict()
            for idx, w_local in enumerate(set_of_w_local):
                for k in w_local.keys():
                    updated_w_glob[k] += (similarity_matrix[user_idx][idx] * w_local[k])

            client_wise_updated_weights[user_idx] = updated_w_glob
            
        return client_wise_updated_weights

    def top_k_average(self, set_of_w_local, similarity_matrix):
        client_wise_updated_weights = {}
        for user_idx in range(self.args.num_users):
            updated_w_glob = get_model(self.args).state_dict()
            top_k_similarity = torch.topk(similarity_matrix[user_idx], self.args.k+1)
            for idx in top_k_similarity[1][1:]:
                for key in set_of_w_local[idx].keys():
                    updated_w_glob[key] += (set_of_w_local[idx][key]/self.args.k)
            client_wise_updated_weights[user_idx] = updated_w_glob
        return client_wise_updated_weights
        
    def train(self, net, body_lr, head_lr, idx=-1, local_eps=None):
        net.train()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
class LocalUpdatePerFedAvg(object):    
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        
        epoch_loss = []
        
        for local_ep in range(self.args.local_ep):
            batch_loss = []
            
            if len(self.ldr_train) / self.args.local_ep == 0:
                num_iter = int(len(self.ldr_train) / self.args.local_ep)
            else:
                num_iter = int(len(self.ldr_train) / self.args.local_ep) + 1
                
            train_loader_iter = iter(self.ldr_train)
            
            for batch_idx in range(num_iter):
                temp_net = copy.deepcopy(list(net.parameters()))
                    
                # Step 1
                for g in optimizer.param_groups:
                    g['lr'] = lr
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                
                
                # Step 2
                for g in optimizer.param_groups:
                    g['lr'] = beta
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                    
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                
                # restore the model parameters to the one before first update
                for old_p, new_p in zip(net.parameters(), temp_net):
                    old_p.data = new_p.data.clone()
                    
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
    def one_sgd_step(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        
        test_loader_iter = iter(self.ldr_train)

        # Step 1
        for g in optimizer.param_groups:
            g['lr'] = lr

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)


        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        # Step 2
        for g in optimizer.param_groups:
            g['lr'] = beta

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)

        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()

        optimizer.step()


        return net.state_dict()

class LocalUpdateFedRep(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': 0.0, 'name': "body"},
                                     {'params': head_params, 'lr': lr, "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
        
        for g in optimizer.param_groups:
            if g['name'] == "body":
                g['lr'] = lr
            elif g['name'] == 'head':
                g['lr'] = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            logits = net(images)

            loss = self.loss_func(logits, labels)
            loss.backward()
            optimizer.step()

        return net.state_dict()

class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (self.args.mu / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 

class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
            
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False, momentum=0.9):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                w_0 = copy.deepcopy(net.state_dict())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if w_ditto is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
