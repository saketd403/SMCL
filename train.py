import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from resnet import ResNet, Bottleneck
from run_epochs import epoch_train, epoch_val


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    parser.add_argument('--lr',default=0.001,type=float,help="learning rate")
    parser.add_argument('--epochs',default=100,type=int,help="Number of training epochs")
    parser.add_argument('--batch_size',default=32,type=int,help="Batch size")
    parser.add_argument('--K',default=1,type=int,help="Number of predictors to be considered")
    parser.add_argument('--M',default=5,type=int,help="Number of members in ensemble")
    parser.add_argument('--dataset_dir',default="./data/cifar10",type=str,help="Number of members in ensemble")
    parser.add_argument('--num_classes',default=10,type=int,help="Number of classes in dataset")
    parser.add_argument('--image_size',default=32,type=int,help="Size of image")
    parser.add_argument('--means',nargs='+',default=[0.4802, 0.4481, 0.3975],type=float,help="channelwise means for normalization")
    parser.add_argument('--stds',nargs='+',default=[0.2302, 0.2265, 0.2262],type=float,help="channelwise std for normalization")
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum")
    parser.add_argument('--weight_decay',default=0.0005,type=float,help="weight decay")
    parser.add_argument('--save_path',default="./save_models/cifar10",type=str,help="Path to save the ensemble weights")

    parser.set_defaults(argument=True)

    return parser.parse_args()


def main():
    args = get_args()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    K = args.K 
    M = args.M 
    root_dir = args.dataset_dir
    num_classes = args.num_classes
    image_size = args.image_size
    means = args.means
    stds = args.stds
    momentum = args.momentum
    weight_decay = args.weight_decay

    transform_train = transforms.Compose([
    transforms.RandomCrop(args.image_size, padding=4),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.RandomAffine(degrees=30,translate =(0.2,0.2),scale=(0.75,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means,stds),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir,"train"),
                                      transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir,"val"),
                                            transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False)

    networks = []
    optimizers = []
    schedulers = []

    for i in range(M):
        net = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes,img_size=image_size)
        net.cuda()
        networks.append(net)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
        optimizers.append(optimizer)
        schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100))

    for epoch in range(epochs):
        
        epoch_train(networks,optimizers,M,K,trainloader)
        test_acc=epoch_val(networks,M,testloader)
        for s in schedulers:
            s.step()
        print("Test accuracy and loss for epoch ",epoch," ",test_acc)
        #torch.save(networks,save_path)
    

if __name__ == "__main__":
    main()


