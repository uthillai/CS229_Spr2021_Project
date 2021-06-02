import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
import cv2
from torchvision import transforms, datasets

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tools.Tester import ModelNetTester
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11") #change for model
parser.add_argument("-num_views", type=int, help="number of views", default=4) #default was 12 with modelnet40
#parser.add_argument("-train_path", type=str, default="modelnet40_images_new_12x/*/train")
parser.add_argument("-train_path", type=str, default="iPhone_images_mvcnn/*/train")
#parser.add_argument("-val_path", type=str, default="modelnet40_images_new_12x/*/test")
parser.add_argument("-val_path", type=str, default="iPhone_images_mvcnn/*/test")
parser.add_argument("-num_classes", type=int, default=15)   #added for number classes 40 for modelnet, 15 for iphones
parser.add_argument("-num_epochs", type=int, default=30)

parser.add_argument("-test_single", type=int, default=1)
parser.add_argument("-test_network", type=str, default="svcnn")
parser.set_defaults(train=False)


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    test_single = args.test_single
    num_classes = args.num_classes  # can now pass in number of classes

    pretraining = not args.no_pretraining
    log_dir = args.name

    image_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                        ])
    #things to change when testing
        # labels list for the number classes
        # label value
        # file path for image
        # model path if testing other models
    CNN1_model_path = "./mvcnn_stage_1_iphone_resnet50_max/mvcnn/model-00006.pth"
    CNN2_model_path = "./mvcnn_stage_2_iphone_resnet50_max/mvcnn/model-00003.pth"

    if test_single:

        labels =['5', '5S', '6', '6 Plus', '6S', '6S Plus', '7', '7 Plus', '8', '8 Plus', 'SE', 'X', 'XR', 'XS', 'XS Max']
        #labels = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser']


        label = '8 Plus' # <Change me only
        file_path1 = "./iPhone_images_mvcnn/{}/test/iphone{}_model1_v026.jpg".format(label,label)
        file_path2 = "./iPhone_images_mvcnn/{}/test/iphone{}_model1_v032.jpg".format(label,label)
        image = cv2.imread(file_path1)
        image1 = cv2.imread(file_path1)
        image2 = cv2.imread(file_path2)

        image1_tensor_mvcnn = image_transform(image1)
        image2_tensor_mvcnn = image_transform(image2)

        image_tensor_mvcnn = image_transform(image)
        C,H,W = image_tensor_mvcnn.shape

        image1_tensor_mvcnn = image1_tensor_mvcnn.reshape(1, C, H, W)
        image2_tensor_mvcnn = image2_tensor_mvcnn.reshape(1, C, H, W)

        image_tensor_mvcnn = image_tensor_mvcnn.reshape(1,C,H,W)
        image_tensor = image_transform(image)

        image3_tensor_mvcnn = torch.cat((image1_tensor_mvcnn, image2_tensor_mvcnn))


    # STAGE 1
    log_dir = args.name + '_stage_1_test'
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=num_classes, pretraining=pretraining, cnn_name=args.cnn_name)
    cnet.load_state_dict(torch.load(CNN1_model_path))

    cnet2 = MVCNN(args.name, cnet, nclasses=num_classes, cnn_name=args.cnn_name, num_views=args.num_views)
    cnet2.load_state_dict(torch.load(CNN2_model_path))

    #optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_models_train = args.num_models * args.num_views

    #train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train,
    #                                 num_views=args.num_views)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
#


   ###################################

    if args.test_network == "svcnn":
        val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,
                                       max_images=10)  # remove max_images = 10 when I need more validation data
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
       #print('num_train_files: ' + str(len(train_dataset.filepaths)))
        print('num_val_files: ' + str(len(val_dataset.filepaths)))
        tester = ModelNetTester(cnet, val_loader, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_classes=15, num_views=1)
        if test_single:
            pred = tester.test_model_with_single_image(image_tensor, labels, label)
            print('labels:', label)
            print(labels[pred.item()]) #print the nth element of labels
        else:
            tester.test_model_with_dataset()

   ######################################


    # STAGE 2
    # log_dir = args.name + '_stage_2_test'
    # create_folder(log_dir)
    # cnet_2 = MVCNN(args.name, cnet, nclasses=num_classes, cnn_name=args.cnn_name, num_views=args.num_views)
    # del cnet
    
    # optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train,
    #                                     num_views=args.num_views)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,
    #                                            num_workers=0)  # shuffle needs to be false! it's done within the trainer
#    print('num_val_files: ' + str(len(val_dataset.filepaths)))
    ##########################################

    if args.test_network == "mvcnn":
        val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views, total_views=8)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        tester = ModelNetTester(cnet2, val_loader, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_classes=15, num_views=args.num_views)
        if test_single:
            pred = tester.test_model_with_image(image3_tensor_mvcnn, labels, label)
            print('labels:', label)
            print('prediction', labels[pred.item()]) #print the nth element of labels

        else:
            tester.test_model_with_dataset()
    ##########################################

