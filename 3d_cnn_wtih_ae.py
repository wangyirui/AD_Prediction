import torch
from autoencoder import AutoEncoder 

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv = nn.Conv3d(1, 410, kernel_size=7, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=7,stride=7)
        self.fc1 = nn.Linear(15*15*15, 800)
        self.fc2 = nn.Linear(800, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, out):
        out = self.conv(out)
        out = self.pool(out)
        out = out.view(1,15*15*15)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

    def load_ae(self, ae):
        cnn.state_dict()['conv.weight'] = ae.state_dict()['encoder.weight'].view(410,1,7,7,7)
        cnn.state_dict()['conv.bias'] = ae.state_dict()['encoder.bias']
