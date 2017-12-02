import argparse
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from AD_Dataset import AD_Dataset
from ResNet import ResNet
from AlexNet import AlexNet


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for JHU CS661 Computer Vision HW3.")

parser.add_argument("--network_type", "--nt", default="AlexNet", choices=["AlexNet", "ResNet"],
                    help="Deep network type. (default=AlexNet)")
parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="best_model.bce_aug",
                    help="Save network weights.")  
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")  
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")               
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need


def main(options):
    # Path configuration
    TRAINING_PATH = 'train.txt'
    TESTING_PATH = 'test.txt'
    IMG_PATH = './Image'

    transformations = transforms.Compose([transforms.Scale((110,110,110)),
                                    transforms.ToTensor()
                                    ])


    dset_train = AD_Dataset(IMG_PATH, TRAINING_PATH, transformations)
    dset_test = AD_Dataset(IMG_PATH, TESTING_PATH, transformations)

    if options.load is None:
        train_loader = DataLoader(dset_train,
                                  batch_size = options.batch_size,
                                  shuffle = True,
                                  num_workers = 4
                                 )
    else:
        train_loader = DataLoader(dset_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=4
                                  )

    test_loader = DataLoader(dset_test,
                             batch_size = options.batch_size,
                             shuffle = False,
                             num_workers = 4
                             )

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    # Training process
    if options.load is None:
        # Initial the model
        if options.network_type == 'AlexNet':
            model = AlexNet()
        else:
            model = ResNet()

        if use_cuda > 0:
            model.cuda()
        else:
            model.cpu()

        # Binary cross-entropy loss
        criterion = torch.nn.NLLLoss()


        optimizer = eval("torch.optim." + options.optimizer)(model.parameters(), options.learning_rate)


        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False)

        # Prepare for label encoding
        last_dev_avg_loss = float("inf")
        best_accuracy = float("-inf")

        # main training loop
        for epoch_i in range(options.epochs):
            logging.info("At {0}-th epoch.".format(epoch_i))
            train_loss = 0.0

            for it, train_data in enumerate(train_loader):
                data_dic = train_data

                if use_cuda:
                    imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda() 
                else:
                    imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])

                print "labels"
                print labels
                integer_encoded = label_encoder.fit_transform(labels)

                # binary encode
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

                train_output = model(imgs)
                loss = criterion(train_output, onehot_encoded)
                train_loss += loss
                logging.debug("loss at batch {0}: {1}".format(it, loss.data[0]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_avg_loss = train_loss / (len(dset_train) / options.batch_size)
            logging.info("Average training loss is {0} at the end of epoch {1}".format(train_avg_loss.data[0], epoch_i))
            
            # validation -- this is a crude esitmation because there might be some paddings at the end
            dev_loss = 0.0
            correct_prediction = 0.0
            for it, test_data in enumerate(test_loader):
                data_dic = test_data

                if use_cuda:
                    imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda() 
                else:
                    imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])


                test_output = model(imgs)
                loss = criterion(test_output, labels)
                dev_loss += loss
            dev_avg_loss = dev_loss / (len(dset_test) / options.batch_size)
            logging.info("Average validation loss is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))
            
            if testing_accuracy > best_accuracy:
                best_accuracy = testing_accuracy
                if options.save is not None:
                    torch.save(model.state_dict(), options.save )
            last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
