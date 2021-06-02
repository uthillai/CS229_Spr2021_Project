import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
import ipdb

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False)
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=15, pretraining=True, cnn_name='vgg11', dataset_name='iphone'):
        super(SVCNN, self).__init__(name)

        if dataset_name == 'iphone':
            self.classnames = ['5', '5S', '6', '6 Plus', '6S', '6S Plus', '7', '7 Plus', '8', '8 Plus', 'SE', 'X', 'XR', 'XS',
                               'XS Max']
        else:
            self.classnames=['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser']

        # self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
        #                    'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
        #                    'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
        #                    'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
        #                    'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']




        self.nclasses = nclasses #print all these
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False) #print shape
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False) #print shape

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=15, cnn_name='vgg11', num_views=12, dataset_name='iphone'):
        super(MVCNN, self).__init__(name)

        if dataset_name == 'iphone':
            self.num_views = num_views
            self.classnames = ['5', '5S', '6', '6 Plus', '6S', '6S Plus', '7', '7 Plus', '8', '8 Plus', 'SE', 'X', 'XR', 'XS',
                               'XS Max']
        else:
            self.num_views =num_views
            self.classnames=['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser']

        # self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
        #                    'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
        #                    'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
        #                    'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
        #                    'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses

        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False)

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x): #max pooling view pooling step
        y = self.net_1(x)
        print('y-shape', y.shape)
        #ipdb.set_trace()
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        #return self.net_2(torch.mean(y, 1).view(y.shape[0], -1)) #for avg pooling
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1)) # take the max features from each feature map of the views per batch