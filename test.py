import argparse
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torchvision

from autoencoder import AutoEncoder 
from AD_3DRandomPatch import AD_3DRandomPatch

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for AutoEncoder")

parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
parser.add_argument("--num_classes", default=2, type=int,
                    help="Number of classes.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")  

def main(options):

    if options.num_classes == 2:
        TRAINING_PATH = 'train_2classes.txt'
    else:
        TRAINING_PATH = 'train.txt'
    IMG_PATH = '/Users/waz/JHU/CV-ADNI/ImageNoSkull'

    dset_train = AD_3DRandomPatch(IMG_PATH, TRAINING_PATH)

    train_loader = DataLoader(dset_train,
                              batch_size = options.batch_size,
                              shuffle = True,
                              num_workers = 4,
                              drop_last = True
                              )

    sparsity = 0.05
    beta = 0.5

    mean_square_loss = nn.MSELoss()
    kl_div_loss = nn.KLDivLoss()

    use_gpu = len(options.gpuid)>=1
    autoencoder = AutoEncoder()


    autoencoder = autoencoder.cpu()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)

    train_loss = 0.
    for epoch in range(options.epochs):
        print("At {0}-th epoch.".format(epoch))
        for i, patches in enumerate(train_loader):
            for b, batch in enumerate(patches):
                batch = Variable(batch) 
                output, mean_activitaion = autoencoder(batch)
                loss1 = mean_square_loss(output, batch)
                loss2 = kl_div_loss(mean_activitaion, Variable(torch.Tensor([sparsity])))
                print "loss1", loss1
                print "loss2", loss2
                loss = loss1 + loss2
                train_loss += loss
                logging.info("batch {0} training loss is : {1:.5f}, {1:.5f}".format(b, loss1.data[0], loss2.data[0]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_avg_loss = train_loss/len(train_loader*1000)
        print("Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data[0], epoch))
    torch.save(model.state_dict(), open("autoencoder_model", 'wb'))

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)