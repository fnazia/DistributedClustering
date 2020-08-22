import argparse
import copy
import numpy as np
import torch
import sys
import os
import pickle, gzip
import traceback
import socket
import datetime
import torch.distributed as dist
from random import Random

import DeepEmbeddedCluster as DEC

def _standardization_param_set(X, Xmeans, Xstds, XstdsFixed, XstdsConst):
        
    if Xmeans is None:
        Xmeans = X.mean(axis = 0)
        Xstds = X.std(axis = 0)
        XstdsConst = Xstds == 0
        XstdsFixed = copy.copy(Xstds)
        XstdsFixed[XstdsConst] = 1
        
    return Xmeans, Xstds, XstdsFixed, XstdsConst
            
def _standardizeX(X, Xmeans, Xstds, XstdsFixed, XstdsConst):
        
    X_standardized = (X - Xmeans) / XstdsFixed
    X_standardized[:, XstdsConst] = 0.0
    return X_standardized
    
def _unstandardizeX(Xs):
        
    return (Xs * Xstds) + Xmeans

def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = '127.0.0.1'   #Add ip-address or hostname
    os.environ['MASTER_PORT'] = '29500'     #Add port number

    # initialize the process group
    dist.init_process_group(backend, rank = int(rank), 
                            world_size = int(world_size),  
                            init_method = 'tcp://127.0.0.1:29499',
                            timeout = datetime.timedelta(weeks = 120))
    torch.manual_seed(42)

if __name__ == '__main__':
    
    try:
    
        print('Running ' + sys.argv[0] + ' ...')
        argv_parser = argparse.ArgumentParser()
        argv_parser.add_argument('--rank', nargs = '?', type = int, default = None)
        argv_parser.add_argument('--world', nargs = '?', type = int, default = None)
        argv_parser.add_argument('--datafile', nargs = '?')
        argv_parser.add_argument('--ae_structure', nargs = '*', type = int, default = [500, 500, 200, 10, 200, 500, 500])
        argv_parser.add_argument('--kernel_stride', nargs = '*', type = list, default = None)
        argv_parser.add_argument('--ae_activation', nargs = '?', default = 'relu')
        argv_parser.add_argument('--clusters', nargs = '?', type = int, default = 10)
        argv_parser.add_argument('--pretrain_epochs', nargs = '?', type = int, default = 200)
        argv_parser.add_argument('--pretrain_lr', nargs = '?', type = float, default = 0.001)
        argv_parser.add_argument('--pretrain_batch_size', nargs = '?', type = int, default = 10000)
        argv_parser.add_argument('--dectrain_epochs', nargs = '?', type = int, default = 200)
        argv_parser.add_argument('--dec_lr', nargs = '?', type = float, default = 0.001)
        argv_parser.add_argument('--dec_batch_size', nargs = '?', type = int, default = 10000)
        argv_parser.add_argument('--gpu', nargs = '?', type = bool, default = False)
        argv_parser.add_argument('--cnn', nargs = '?', type = bool, default = False)
        argv_parser.add_argument('--dist', nargs = '?', type = bool, default = False)
        argv_parser.add_argument('--pretrain_weights_path', type=str, default='./ae_pretrained')
        argv_parser.add_argument('--gamma', type=float, default=0.1)
        argv_parser.add_argument('--target_update_epochs', nargs = '?', type = int, default = 10)


        args = argv_parser.parse_args()

        activation_dict = {'relu': torch.nn.ReLU(True), 'tanh': torch.nn.Tanh(), 'sigmoid': torch.nn.Sigmoid()}
        ae_activation = activation_dict[args.ae_activation]

        args.gpu = True if args.gpu and torch.cuda.is_available() else False
        device = torch.device("cuda" if args.gpu else "cpu")

        with gzip.open(args.datafile, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        Xtrain = train_set[0]
        Ttrain = train_set[1]

        #Xvalid = valid_set[0]
        #Tvalid = valid_set[1]

        #Xtest = test_set[0]
        #Ttest = test_set[1]

        Xmeans = None
        Xstds = None
        XstdsFixed = None
        XstdsConst = None

        n_channels = None
        image_size = None
        kernel_stride = None

        if args.cnn:
            Xtrain = Xtrain.reshape(-1, 1, 28, 28)
            Ttrain = Ttrain.reshape(-1, 1)

            #Xvalid = Xvalid.reshape(-1, 1, 28, 28)
            #Tvalid = Tvalid.reshape(-1, 1)

            #Xtest = Xtest.reshape(-1, 1, 28, 28)
            #Ttest = Ttest.reshape(-1, 1)

            n_channels = Xtrain.shape[1]
            image_size = Xtrain.shape[2]

            kernel_stride = [list(map(int, item)) for item in args.kernel_stride]

        Xtrain = torch.tensor(Xtrain)
        Xmeans, Xstds, XstdsFixed, XstdsConst = _standardization_param_set(Xtrain, Xmeans, Xstds, XstdsFixed, XstdsConst)
        Xtrain = _standardizeX(Xtrain, Xmeans, Xstds, XstdsFixed, XstdsConst)

        decmodel = DEC.DeepEmbeddedCluster(features = Xtrain.shape[1], 
                                           auto_struct = args.ae_structure, 
                                           auto_activation = ae_activation, 
                                           n_clusters = args.clusters, 
                                           alpha = 1.0, 
                                           n_channels = n_channels, 
                                           image_size = image_size, 
                                           kernel_size_and_stride = kernel_stride, 
                                           cnn = args.cnn, 
                                           pretrain_weights_path = args.pretrain_weights_path,
                                           dec_batch_size = args.dec_batch_size, 
                                           dectrain_epochs = args.dectrain_epochs, 
                                           dec_lr = args.dec_lr,
                                           pretrain_batch_size = args.pretrain_batch_size, 
                                           pretrain_epochs = args.pretrain_epochs, 
                                           pretrain_lr = args.pretrain_lr, 
                                           gamma = args.gamma, 
                                           target_update_epochs = args.target_update_epochs, 
                                           device = device,
                                           distrain = args.dist)

        if args.dist:
            backend = "nccl" if args.gpu else "gloo"
            print(args.gpu, device, backend)
            setup(args.rank, args.world, backend)
            print(socket.gethostname()+": Setup completed.")

        Xtrain = Xtrain.to(device)
        decmodel.train(Xtrain.type(torch.FloatTensor))
        
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
    
