import argparse
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image

import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from collections import Counter

from custom_transform2D import CustomResize
from custom_transform2D import CustomToTensor

from AD_Dataset import AD_Dataset
from AD_Standard_2DSlicesData import AD_Standard_2DSlicesData
from AD_Standard_2DRandomSlicesData import AD_Standard_2DRandomSlicesData
from AD_Standard_2DTestingSlices import AD_Standard_2DTestingSlices

from AlexNet2D import alexnet

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for JHU CS661 Computer Vision HW3.")

parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="AlexNet",
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
    TRAINING_PATH = 'train_2classes.txt'
    TESTING_PATH = 'test_2classes.txt'
    IMG_PATH = './Image'

    trg_size = (224, 224)

    transformations = transforms.Compose([CustomResize(trg_size),
                                          CustomToTensor()
                                        ])
    dset_train = AD_Standard_2DRandomSlicesData(IMG_PATH, TRAINING_PATH, transformations)
    dset_test = AD_Standard_2DSlicesData(IMG_PATH, TESTING_PATH, transformations)

    # Use argument load to distinguish training and testing
    if options.load is None:
        train_loader = DataLoader(dset_train,
                                  batch_size=options.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True
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
                             batch_size=options.batch_size,
                             shuffle=False,
                             num_workers=4,
                             drop_last=True
                             )

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        cuda.set_device(options.gpuid[0])


    # Initial the model
    model = alexnet(pretrained=True)
    # model.load_state_dict(torch.load(options.load))

    if use_cuda > 0:
        model.cuda()
    else:
        model.cpu()

    # Binary cross-entropy loss
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()

    lr = options.learning_rate
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)

    best_accuracy = float("-inf")

    train_loss_f = open("train_loss.txt", "w")
    test_acu_f = open("test_accuracy.txt", "w")

    for epoch_i in range(options.epochs):

        logging.info("At {0}-th epoch.".format(epoch_i))
        train_loss, correct_cnt = train(model, train_loader, use_cuda, criterion, optimizer, train_loss_f)
        # each instance in one batch has 3 views
        train_avg_loss = train_loss / (len(dset_train) * 3 / options.batch_size)
        train_avg_acu = float(correct_cnt) / (len(dset_train) * 3)
        logging.info(
            "Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data[0], epoch_i))
        logging.info("Average training accuracy is {0:.5f} at the end of epoch {1}".format(train_avg_acu, epoch_i))


        correct_cnt = validate(model, test_loader, use_cuda, criterion)
        dev_avg_acu = float(correct_cnt) / len(dset_test)
        logging.info("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(dev_avg_acu, epoch_i))

        # write validation accuracy to file
        test_acu_f.write("{0:.5f}\n".format(dev_avg_acu))

        if dev_avg_acu > best_accuracy:
            best_accuracy = dev_avg_acu
            torch.save(model.state_dict(), open(options.save, 'wb'))

    train_loss_f.close()
    test_acu_f.close()

def train(model, train_loader, use_cuda, criterion, optimizer, train_loss_f):
    # main training loop
    train_loss = 0.0
    correct_cnt = 0.0
    model.train()
    for it, train_data in enumerate(train_loader):
        for data_dic in train_data:
            if use_cuda:
                imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda()
            else:
                imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])
            integer_encoded = labels.data.cpu().numpy()
            # target should be LongTensor in loss function
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
            if use_cuda:
                ground_truth = ground_truth.cuda()
            train_output = model(imgs)
            _, predict = train_output.topk(1)
            loss = criterion(train_output, ground_truth)
            train_loss += loss
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            correct_cnt += correct_this_batch
            accuracy = float(correct_this_batch) / len(ground_truth)
            logging.info("batch {0} training loss is : {1:.5f}".format(it, loss.data[0]))
            logging.info("batch {0} training accuracy is : {1:.5f}".format(it, accuracy))

            # write the training loss to file
            train_loss_f.write("{0:.5f}\n".format(loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return train_loss, correct_cnt



def validate(model, test_loader, use_cuda, criterion):
    # validation -- this is a crude estimation because there might be some paddings at the end
    correct_cnt = 0.0
    model.eval()
    for it, test_data in enumerate(test_loader):
        vote = []
        for data_dic in test_data:
            if use_cuda:
                imgs, labels = Variable(data_dic['image'], volatile=True).cuda(), Variable(data_dic['label'],
                                                                                           volatile=True).cuda()
            else:
                imgs, labels = Variable(data_dic['image'], volatile=True), Variable(data_dic['label'],
                                                                                    volatile=True)
            test_output = model(imgs)
            _, predict = test_output.topk(1)
            vote.append(predict)

        vote = torch.cat(vote, 1)
        final_vote, _ = torch.mode(vote, 1)
        ground_truth = test_data[0]['label']
        correct_this_batch = (final_vote.cpu().data == ground_truth).sum()
        correct_cnt += correct_this_batch
        accuracy = float(correct_this_batch) / len(ground_truth)

        logging.info("batch {0} dev accuracy is : {1:.5f}".format(it, accuracy))

    return correct_cnt




def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)