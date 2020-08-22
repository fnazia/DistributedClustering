import torch
import numpy as np
import copy
import time
from sklearn.cluster import KMeans
from math import ceil

import AutoEncoder as AE
import CustomModel as CM
import DistributedTraining as DT

class DeepEmbeddedCluster():
    
    def __init__(self, features, auto_struct, auto_activation, n_clusters, alpha, n_channels, image_size, 
                kernel_size_and_stride, cnn, pretrain_weights_path, dec_batch_size, dectrain_epochs, dec_lr, 
                pretrain_batch_size, pretrain_epochs, pretrain_lr, gamma, target_update_epochs, 
                device, distrain = False):
    
        self.distrain = distrain
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.clusters = n_clusters
        self.pretrain_weights_path = pretrain_weights_path
        self.dec_batch_size = dec_batch_size
        self.dectrain_epochs = dectrain_epochs
        self.dec_lr = dec_lr
        self.n_features = features #X.shape[1]
        self.target_update_epochs = target_update_epochs
        
        self.autoencoder = AE.AutoEncoder(auto_struct, auto_activation, self.n_features, n_channels, image_size, 
                              kernel_size_and_stride, pretrain_batch_size, pretrain_epochs, pretrain_lr, 
                              pretrain_weights_path, cnn, distrain = self.distrain)
        
        hidden_features = self.autoencoder.autonet.hidden.out_features
        
        self.model = CM.CustomModel(self.autoencoder, n_clusters, hidden_features, alpha).to(self.device)
        
        if self.distrain:
            self.dt = DT.DistributedTraining()
        
    def train(self, Xtr):
    
        time_start = time.time()
        delta_target = 0.001
        error_trace = []

        self.pretrain(Xtr, self.model.aemodel)

        with torch.no_grad():
            hidden_out = self.model.aemodel.get_latent_output(Xtr)

        kmeans = KMeans(n_clusters = self.clusters, n_init = 20)
        T_pred = kmeans.fit_predict(hidden_out.data.cpu()) #.numpy())

        hidden_out = None
        
        T_pred = torch.tensor(T_pred)
        T_pred_prev = copy.deepcopy(T_pred).to(self.device)
        self.model.custom_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        loss_func = torch.nn.MSELoss()
        loss_func_kl = torch.nn.KLDivLoss(reduction = 'batchmean')

        if self.distrain:
             self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters = True) #.float()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.dec_lr)
        self.model.train()

        for epoch in range(self.dectrain_epochs):
            epoch_error = 0.0
            if epoch % self.target_update_epochs == 0:
                with torch.no_grad():
                    _, q = self.model(Xtr)
                q = q.data
                p = self.target_distribution(q)

                T_pred = q.argmax(1)
                delta_current = torch.sum(T_pred != T_pred_prev).type(torch.float32) / T_pred.shape[0]
                T_pred_prev = T_pred

                if epoch > 0 and delta_current < delta_target:   
                    break
            inputlist = []
            for i, c in zip(Xtr, p):
                inputlist.append((i, c))
                
            if self.distrain:
                trainloader, bsz = self.dt.partition_dataset(inputlist, self.dec_batch_size)
            else:
                trainloader = torch.utils.data.DataLoader(inputlist, batch_size = self.dec_batch_size, 
                                                              shuffle = False)
            for batch in trainloader:
                Xb, pb = batch
                X_out, qb = self.model(Xb)
                reconst_error = loss_func(X_out, Xb)
                kl_error = loss_func_kl(qb.log(), pb)
                error = self.gamma * kl_error + reconst_error
                epoch_error += error.item()
                optimizer.zero_grad()
                error.backward()
                if self.distrain:
                    self.dt.average_gradients(self.model)
                optimizer.step()
            error_trace.append(epoch_error/len(trainloader))
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1} error {error_trace[-1]:.5f}')
        
        #train_pred, qn = self.test(Xtr)
        self.model.eval()
        _, qt = self.model(Xtr)
        qt = qt.data
        qn = qt.cpu().numpy()
        train_pred = qn.argmax(1)

        training_time = time.time() - time_start
        
        #Save prediction, final layer output, error trace, and required training time
        np.savez('trainresult', clpred = train_pred, qout = qn, err = error_trace, traintime = training_time)
        
        #print(f'Training accuracy: {normalized_mutual_info_score(Ttrain.flatten(), train_pred)}')
        print('All done.')
 
        
    def pretrain(self, Xtr, aem):
        
        self.model.aemodel.ae_train(Xtr, aem)
        
    def test(self, X):
        
        self.model.eval()
        _, qt = self.model(Xtr)
        qt = qt.data
        qn = qt.cpu().numpy()
        train_pred = qn.argmax(1)
        return train_pred, qn
    
    def target_distribution(self, q):
    
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

