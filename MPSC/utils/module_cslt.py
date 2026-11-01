import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.utils import init_weights


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_scale_factor, projection_size, init_method='He', activation='relu'):
        super().__init__()

        mlp_hidden_size = round(mlp_scale_factor * in_channels)
        if activation == 'relu':
            non_linear_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky relu':
            non_linear_layer = nn.LeakyReLU(inplace=True)
        elif activation == 'tanh':
            non_linear_layer = nn.Tanh()
        else:
            raise AssertionError(f'{activation} is not supported yet.')

        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            non_linear_layer,
            nn.Linear(mlp_hidden_size, projection_size)
        )
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)

class CSGRLClassifier(nn.Module):

    def __init__(self, inchannels, num_class, req_rx = True, ae_latent=64):
        super().__init__()
        ae_hidden = []
        self.class_aes = []
        for i in range(num_class):
            ae = AutoEncoder(inchannels, ae_hidden, ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.useL1 = True
        self.req_rx = req_rx

        self.reduction = -1
        self.reduction *= 0.01

    def ae_error(self, rc, x):
        if self.useL1:
            # return torch.sum(torch.abs(rc-x) * self.reduction,dim=1,keepdim=True)
            return torch.norm(rc - x, p=1, dim=1, keepdim=True) * self.reduction
        else:
            return torch.norm(rc - x, p=2, dim=1, keepdim=True) ** 2 * self.reduction

    clip_len = 100

    def forward(self, x):
        cls_ers = []
        lts = []
        for i in range(len(self.class_aes)):
            rc, lt = self.class_aes[i](x)
            cls_er = self.ae_error(rc, x)
            if CSGRLClassifier.clip_len > 0:
                cls_er = torch.clamp(cls_er, -CSGRLClassifier.clip_len, CSGRLClassifier.clip_len)
            cls_ers.append(cls_er)
            lts.append(lt)
        logits = torch.cat(cls_ers, dim=1)
        logits = self.avg_pool(logits).view(x.size(0), -1)
        if self.req_rx:
            idxs = torch.argmax(logits, dim=1)
            lt = torch.stack([lts[idx][row] for row, idx in zip(torch.arange(0, len(idxs)), idxs)])
            lt = self.avg_pool(lt).view(x.size(0), -1)
            return logits, lt
        else:
            return logits


class AutoEncoder(nn.Module):

    def __init__(self, inchannel, hidden_layers, latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan
        if latent_chan > 0:
            self.encode_convs = []
            self.decode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel, h, )
                dcv = layer_block(h, inchannel, use_activation=i != 0)
                inchannel = h
                self.encode_convs.append(ecv)
                self.decode_convs.append(dcv)
            self.encode_convs = nn.ModuleList(self.encode_convs)
            self.decode_convs.reverse()
            self.decode_convs = nn.ModuleList(self.decode_convs)
            self.latent_conv = layer_block(inchannel, latent_chan)
            self.latent_deconv = layer_block(latent_chan, inchannel, use_activation=(len(hidden_layers) > 0))
        else:
            self.center = nn.Parameter(torch.rand([inchannel, 1, 1]), True)

    def forward(self, x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            latent = self.latent_conv(output)
            output = self.latent_deconv(latent)
            for cv in self.decode_convs:
                output = cv(output)
            return output, latent
        else:
            return self.center, self.center

def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False)
    return res

def byx_getscores(x, scores, type, y=None):
    if type == 'max':
        idxs = []
        for k in range(len(y)):
            idx = torch.argsort(x[k,:])
            if idx[-1] == y[k]:
                idxs.append(idx[-2])
            else:
                idxs.append(idx[-1])
        idxs = torch.tensor(idxs)
    elif type == 'y':
        idxs = x
    outputs = torch.stack([scores[row][idx] for row, idx in zip(torch.arange(0, len(idxs)), idxs)])
    # outputs = nn.AdaptiveAvgPool2d((1, 1))(max_scores).view(x.size(0), -1)
    return outputs