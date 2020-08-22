# Improved Deep Embedded Clustering Distributed Training
This is a pytorch distributed training implementation of Improved Deep Embedded Clustering. Both fully-connected autoencoder and convolutional autoencoder can be used. 

It is required to put the correct hostname or IP-address and port number in DIDEC.py before running the program with necessary command line arguments. An example of command line run of the program is -

python3 DIDEC.py --rank 0 --world 5 --datafile mnist.pkl.gz --ae_structure 10 5 3 10 3 5 10 --kernel_stride 31 31 31 31 31 31 31  --ae_activation relu --clusters 10 --pretrain_epochs 50 --dectrain_epochs 100 --cnn True --gpu True --dist True

DIDEC.py        --- file containing main function
rank            --- rank number of the machine
world           --- total number of machines involved in distributed training
datafile        --- path of the file containing the data
ae_structure    --- autoencoder structure in order (each number represents number of neurons in that layer. Here first AE layer contains 10 neurons.)
kernel_stride   --- kernel size and stride size (each number represents kernel size and stride size in that layer. Here first AE layer's kernel size is 3 and stride is 1; hence the number '31')
ae_activation   --- autoencoder activation function (tanh, relu, and sigmoid can be used)
clusters        --- number of clusters to be created
pretrain_epochs --- number of epochs in autoencoder pretraining
dectrain_epochs --- number of epochs in DEC training
cnn             --- flag to turn on Convolutional autoencoder (default is set to False and uses fully-connected autoencoder)
gpu             --- flag to turn on training on gpu (default is False and the training will run on CPU)
dist            --- flag to turn on distributed training (default is False and trains the model on single machine)
