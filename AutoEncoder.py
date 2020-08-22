import torch
import numpy as np
import copy
import time
from math import ceil

import DistributedTraining as DT

class AutoEncoder(torch.nn.Module):
    
    def __init__(self, auto_struct, auto_activation, n_features, n_channels, image_size, 
                 kernel_size_and_stride, pretrain_batch_size, pretrain_epochs, 
                 pretrain_lr, pretrain_weights_path, cnn, distrain):
        
        super().__init__()
        self.distrain = distrain
        
        self.n_channels_in_image = n_channels
        self.image_size = image_size
        
        self.pretrain_batch_size = pretrain_batch_size
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_weights_path = pretrain_weights_path
        
        self.autonet = torch.nn.Sequential()
        self.cnn = cnn
        self.re_shape = []
        
        if self.cnn:
            self.cnn_autoencoder(auto_struct, kernel_size_and_stride, auto_activation, n_channels, image_size)
        else:
            self.autoencoder(n_features, auto_struct, auto_activation)
            
        if self.distrain:
            self.dt = DT.DistributedTraining()
        
    def autoencoder(self, input_dim, auto_struct, activation):
        
        feature_in = input_dim
        auto_length = len(auto_struct)
        for i in range(auto_length):
            if i < int(auto_length / 2):
                self.autonet.add_module(f'encoder_{i+1}', torch.nn.Linear(feature_in, auto_struct[i]))
                self.autonet.add_module(f'encoder_{i+1}_act', activation)
            elif i == int(auto_length / 2):
                self.autonet.add_module('hidden', torch.nn.Linear(feature_in, auto_struct[i]))
            else:
                self.autonet.add_module(f'decoder_{auto_length - i}', torch.nn.Linear(feature_in, auto_struct[i]))
                self.autonet.add_module(f'decoder_{auto_length - i}_act', activation)
            feature_in = auto_struct[i]
        self.autonet.add_module(f'decoder_out', torch.nn.Linear(feature_in, input_dim))
        
                
    def cnn_autoencoder(self, auto_struct, kernels_size_and_stride, activation, n_channels_in_image, image_size):
        
        auto_length = len(auto_struct)
        n_units_previous = n_channels_in_image
        output_size_previous = image_size
        i = 0
    
        for (n_units, kernel) in zip(auto_struct, kernels_size_and_stride):
            kernel_size, kernel_stride = kernel
            if i < int(auto_length / 2):
                self.autonet.add_module(f'encoder_{i+1}', torch.nn.Conv2d(n_units_previous, n_units,
                                                                     kernel_size, kernel_stride))
                self.autonet.add_module(f'encoder_{i+1}_act', activation)
                
            elif i == int(auto_length / 2):
                self.autonet.add_module('flatten', torch.nn.Flatten())
                fc_units = output_size_previous ** 2 * n_units_previous
                self.autonet.add_module('hidden', torch.nn.Linear(fc_units, n_units))
                self.autonet.add_module('fc_layer', torch.nn.Linear(n_units, fc_units))
                self.autonet.add_module('fc_layer_act', activation)
                i += 1
                continue
                
            elif i == int(auto_length / 2) + 1:
                i += 1
                self.re_shape.append(n_units)
                self.re_shape.append(output_size_previous)
                continue
                
            else:
                self.autonet.add_module(f'decoder_{auto_length - i}', torch.nn.ConvTranspose2d(n_units_previous, n_units,
                                                                     kernel_size, kernel_stride))
                self.autonet.add_module(f'decoder_{auto_length - i}_act', activation)
            output_size_previous = (output_size_previous - kernel_size) // kernel_stride + 1
            n_units_previous = n_units
            i += 1
        self.autonet.add_module(f'decoder_out', torch.nn.ConvTranspose2d(n_units_previous, n_channels_in_image, 3))
                
               
    def forward(self, X):
        
        hidden_layer_index = int(len(self.autonet) / 2) + 1 if self.cnn else int(len(self.autonet) / 2)
        X = self.autonet[:hidden_layer_index](X)
        hidden_out = copy.copy(X)
        if self.cnn:
            X_out = self.autonet[hidden_layer_index:hidden_layer_index+2](X)
            X_out = X_out.reshape(-1, self.re_shape[0], self.re_shape[1], self.re_shape[1])
            X_out = self.autonet[hidden_layer_index+2:](X_out)
        else:
            X_out = self.autonet[hidden_layer_index:](X)
        return X_out, hidden_out
    
    def ae_train(self, Xtr, ae_model):
        
        if self.distrain:
            aemodel = torch.nn.parallel.DistributedDataParallel(ae_model).float()
            trainloader, bsz = self.dt.partition_dataset(Xtr, self.pretrain_batch_size)
        else:
            aemodel = ae_model
            trainloader = torch.utils.data.DataLoader(Xtr, batch_size = self.pretrain_batch_size, shuffle = True)

        optimizer = torch.optim.Adam(aemodel.parameters(), lr = self.pretrain_lr)
        loss_func = torch.nn.MSELoss()

        error_trace = []

        for epoch in range(self.pretrain_epochs):
            epoch_error = 0.0
            for batch in trainloader:
                Xb = batch 
                optimizer.zero_grad()
                X_pred, _ = aemodel(Xb)
                error = loss_func(X_pred, Xb)
                epoch_error += error.item()
                error.backward()
                if self.distrain:
                    self.dt.average_gradients(aemodel)
                optimizer.step()
            error_trace.append(epoch_error/len(trainloader)) 
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1} error {error_trace[-1]:.5f}')
        torch.save(aemodel.state_dict(), self.pretrain_weights_path)
            
    def get_latent_output(self, X):
        self.autonet.eval()
        hidden_layer_index = int(len(self.autonet) / 2) + 1 if self.cnn else int(len(self.autonet) / 2)
        Xl = self.autonet[:hidden_layer_index](X)
        return Xl
