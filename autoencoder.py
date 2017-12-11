import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(343, 410)
        self.sparsify = nn.Sigmoid()
        self.decoder = nn.Linear(410, 343)

    def forward(self, out):
        out = out.view(-1, 343)
        out = self.encoder(out)
        out = self.sparsify(out)
        s_ = out
        out = self.decoder(out)
        return out, s_
