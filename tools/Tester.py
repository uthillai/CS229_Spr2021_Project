import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import ipdb


class ModelNetTester(object):

    def __init__(self, model, val_loader, loss_fn,
                 model_name, log_dir, num_classes=15, num_views=12):

        #self.optimizer = optimizer
        self.model = model
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.num_classes = num_classes #add in number of classes

        self.model
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def test_model_with_dataset(self, num_classes=15):

        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(num_classes)
        samples_class = np.zeros(num_classes)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []
        confusion_matrix = []

        for _, data in tqdm(enumerate(self.val_loader, 0)):

            #ipdb.set_trace()
            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W)
            else:#'svcnn'
                in_data = Variable(data[1])
            target = Variable(data[0])

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            confusion_matrix.append((target, pred)) # create tuple for confusion matrix

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        confusion_mat = np.zeros([num_classes, num_classes])
        for i, j in confusion_matrix:
            confusion_mat[i,j] += 1
        print('Confusion Matrix: ', confusion_mat)


        print ('Total # of test models: ', all_points)
        # val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        # print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc



    def test_model_with_single_image(self, data, labels, label):
       
        #test single image/view

        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)



        if self.model_name == 'mvcnn':
            print('data.size():', data.size())
            V,C,H,W = data.size()
            data = data.reshape(1,V,C,H,W)
            print(data.shape)
            in_data = Variable(data).view(-1,C,H,W)
        else:#'svcnn'
            print('data shape', data.shape)
            C,H,W = data.shape
            in_data = Variable(data).view(1,C,H,W)
        #target = Variable(label)

        out_data = self.model(in_data)
        pred = torch.max(out_data, 1)[1]
        #print(out_data, labels)
        print('prediction', labels[pred], 'confidence', '/n', out_data)
        #all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
        results = labels[pred] == label
        print("Correct prediction?: ", results)

        return pred