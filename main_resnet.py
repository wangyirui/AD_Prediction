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

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import random

from custom_transform import CustomResize
from custom_transform import CustomToTensor

from AD_Dataset import AD_Dataset
from AD_2DSlicesData import AD_2DSlicesData

from AlexNet2D import alexnet
from AlexNet3D import AlexNet

import ResNet2D
import ResNet3D


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for JHU CS661 Computer Vision HW3.")

parser.add_argument("--network_type", "--nt", default="AlexNet2D", choices=["AlexNet2D", "AlexNet3D", "ResNet2D", "ResNet3D"],
                    help="Deep network type. (default=AlexNet)")
parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="best_model",
                    help="Save network weights.")  
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")  
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
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

    if options.network_type == 'AlexNet3D':
        trg_size = (224, 224, 224)
    elif options.network_type == 'AlexNet2D':
        trg_size = (224, 224)
    elif options.network_type == 'ResNet3D':
        trg_size = (110, 110, 110)
    elif options.network_type == 'ResNet2D':
        trg_size = (224, 224)
    
    if options.network_type == "AlexNet3D" or "ResNet3D":
        transformations = transforms.Compose([CustomResize(options.network_type, trg_size),
                                              CustomToTensor(options.network_type)
                                        ])
        dset_train = AD_2DSlicesData(IMG_PATH, TRAINING_PATH, transformations)
        dset_test = AD_2DSlicesData(IMG_PATH, TESTING_PATH, transformations)

    elif options.network_type == 'AlexNet2D' or "ResNet2D":
        transformations = transforms.Compose([transforms.Resize(trg_size, Image.BICUBIC),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()
                                              ])
        dset_train = AD_2DSlicesData(IMG_PATH, TRAINING_PATH, transformations)
        dset_test = AD_2DSlicesData(IMG_PATH, TESTING_PATH, transformations)

    # Use argument load to distinguish training and testing
    if options.load is None:
        train_loader = DataLoader(dset_train,
                                  batch_size = options.batch_size,
                                  shuffle = True,
                                  num_workers = 4,
                                  drop_last = True
                                 )
    else:
        # Only shuffle the data when doing training
        train_loader = DataLoader(dset_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=True
                                  )

    test_loader = DataLoader(dset_test,
                             batch_size = options.batch_size,
                             shuffle = False,
                             num_workers = 4,
                             drop_last=True
                             )

    use_cuda = (len(options.gpuid) >= 1)
    # if options.gpuid:
    #     cuda.set_device(options.gpuid[0])

    # Training process
    if options.load is None:
        # Initial the model
        if options.network_type == 'AlexNet3D':
            model = AlexNet()
        elif options.network_type == 'AlexNet2D':
            model = alexnet(pretrained=True)
        elif options.network_type == 'ResNet2D':
            model = ResNet2D.resnet152(pretrained=True)
        elif options.network_type == 'ResNet3D':
            model = ResNet3D.ResNet()

        if use_cuda > 0:
            model = nn.DataParallel(model, device_ids=options.gpuid).cuda()
        else:
            model.cpu()

        # Binary cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()

        lr = options.learning_rate
        optimizer = eval("torch.optim." + options.optimizer)(model.parameters(), lr,
                                                             #momentum=options.momentum,
                                                             weight_decay=options.weight_decay)
        # Prepare for label encoding
        last_dev_avg_loss = float("inf")
        best_accuracy = float("-inf")

        # main training loop
        for epoch_i in range(options.epochs):
            logging.info("At {0}-th epoch.".format(epoch_i))
            train_loss = 0.0
            correct_cnt = 0.0
            model.train()
            for it, train_data in enumerate(train_loader):
                data_dic = train_data

                if use_cuda:
                    imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda() 
                else:
                    imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])

                # add channel dimension: (batch_size, D, H ,W) to (batch_size, 1, D, H ,W)
                # since 3D convolution requires 5D tensors
                img_input = imgs#.unsqueeze(1)

                integer_encoded = labels.data.cpu().numpy()
                # target should be LongTensor in loss function
                ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
                if use_cuda:
                    ground_truth = ground_truth.cuda()
                train_output = model(img_input)
                train_prob_predict = F.softmax(train_output, dim=1)
                _, predict = train_prob_predict.topk(1)
                loss = criterion(train_output, ground_truth)

                train_loss += loss
                correct_this_batch = (predict.squeeze(1) == ground_truth).sum()
                correct_cnt += correct_this_batch
                accuracy = float(correct_this_batch) / len(ground_truth)
                logging.info("batch {0} training loss is : {1:.5f}".format(it, loss.data[0]))
                logging.info("batch {0} training accuracy is : {1:.5f}".format(it, accuracy))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_avg_loss = train_loss / (len(dset_train) / options.batch_size)
            train_avg_acu = float(correct_cnt) / len(dset_train)
            logging.info("Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data[0], epoch_i))
            logging.info("Average training accuracy is {0:.5f} at the end of epoch {1}".format(train_avg_acu, epoch_i))
            
            # validation -- this is a crude esitmation because there might be some paddings at the end
            dev_loss = 0.0
            correct_cnt = 0.0
            model.eval()
            for it, test_data in enumerate(test_loader):
                data_dic = test_data

                if use_cuda:
                    imgs, labels = Variable(data_dic['image'], volatile=True).cuda(), Variable(data_dic['label'], volatile=True).cuda() 
                else:
                    imgs, labels = Variable(data_dic['image'], volatile=True), Variable(data_dic['label'], volatile=True)

                img_input = imgs#.unsqueeze(1)
                integer_encoded = labels.data.cpu().numpy()
                ground_truth = Variable(torch.from_numpy(integer_encoded), volatile=True).long()
                if use_cuda:
                    ground_truth = ground_truth.cuda()
                test_output = model(img_input)
                test_prob_predict = F.softmax(test_output, dim=1)
                _, predict = test_prob_predict.topk(1)
                loss = criterion(test_output, ground_truth)
                dev_loss += loss
                correct_this_batch = (predict.squeeze(1) == ground_truth).sum()
                correct_cnt += (predict.squeeze(1) == ground_truth).sum()
                accuracy = float(correct_this_batch) / len(ground_truth)
                logging.info("batch {0} dev loss is : {1:.5f}".format(it, loss.data[0]))
                logging.info("batch {0} dev accuracy is : {1:.5f}".format(it, accuracy))

            dev_avg_loss = dev_loss / (len(dset_test) / options.batch_size)
            dev_avg_acu = float(correct_cnt) / len(dset_test)
            logging.info("Average validation loss is {0:.5f} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))
            logging.info("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(dev_avg_acu, epoch_i))

            torch.save(model.state_dict(), open(options.save + ".nll_{0:.3f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'))

            last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
