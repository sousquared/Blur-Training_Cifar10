import sys 
sys.path.append('../')
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import dataloader, GaussianBlur_images, save_checkpoint, , accuracy
from models import AlexNetCifar10

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--exp-name', '-n', type=str, default='',
                    help='Experiment name.')
parser.add_argument('--normal', action='store_true', default=False,
                    help='Normal training mode (w/o blurring images).')
parser.add_argument('--kernel-size', '-k', type=int, nargs=2, default=(3,3),
                    help='Kernel size of Gaussian Blur.')
parser.add_argument('--sigma', '-s', type=float, default=1,
                    help='Sigma of Gaussian Blur.')
parser.add_argument('--epochs', '-e', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', '-b', type=int, default=64,
                    help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight-decay', '-w', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

def main():
    args = parser.parse_args()
    if args.exp_name == '':
        print('ERROR: USE \'--exp-name\' or \'-n\' option to define this experiment\'s name.')
        sys.exit()

    # directories settings
    os.makedirs('../logs/outputs', exist_ok=True)
    os.makedirs('../logs/models/{}'.format(args.exp_name), exist_ok=True)

    OUTPUT_PATH = '../logs/outputs/{}.log'.format(args.exp_name)
    MODEL_PATH = '../logs/models/{}/'.format(args.exp_name)


    if os.path.exists(OUTPUT_PATH):
        print('ERROR: This \'--exp-name\' is already used. Use another name for this experiment.')
        sys.exit()

    # recording outputs
    sys.stdout = open(OUTPUT_PATH, 'w')
    sys.stderr = open(OUTPUT_PATH, 'a')

    # tensorboardX
    writer = SummaryWriter(log_dir='../logs/tb/{}'.format(args.exp_name))

    # cuda settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('device: {}'.format(device))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # data settings
    trainloader, testloader, _ = dataloader(batch_size=args.batch_size)

    # Model, Criterion, Optimizer
    net = AlexNetCifar10().to(device)
    criterion = nn.CrossEntropyLoss()to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # print settings
    print('='*5 + ' settings ' + '='*5)
    print('TRAINING MODE: {}'.format('Blur' if not args.normal else 'Normal'))
    if not args.normal:
        print('Kernel-size: {}'.format(tuple(args.kernel_size)))
        print('Sigma: {}'.format(args.sigma))
    print('Random seed: {}'.format(args.seed))
    print('Epochs: {}'.format(args.epochs))
    print('Learning rater: {}'.format(args.lr))
    print('Weight_decay: {}'.format(args.weight_decay))
    print()
    print(net)
    print('='*20)
    print()

    # training
    print('Start Training...')
    train_time = time.time()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        # ===== train mode =====
        train_acc = AverageMeter('train_acc', ':6.2f')
        train_loss = AverageMeter('train_loss', ':.4e')
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0], data[1].to(device)

            # Blur images
            if not args.normal:
                inputs = GaussianBlur_images(inputs, \
                                             tuple(args.kernel_size), args.sigma)  
            
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + record
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            acc1 = accuracy(outputs, labels, topk=(1,))
            train_loss.update(loss.item(), inputs.size())
            train_acc.update(acc1[0], inputs.size())
            
            # backward + optimize
            loss.backward()
            optimizer.step()

        # record the values in tensorboard
        writer.add_scalar('loss/train', train_loss.avg , epoch + 1)  # average loss
        writer.add_scalar('acc/train', train_acc.avg , epoch + 1)  # average acc

        # ===== val mode =====
        val_acc = AverageMeter('val_acc', ':6.2f')
        val_loss = AverageMeter('val_loss', ':.4e')
        net.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                """
                if not args.normal:
                    inputs = GaussianBlur_images(inputs.cpu(), \
                                                 tuple(args.kernel_size), args.sigma) 
                    inputs = inputs.to(device)
                """
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                acc1 = accuracy(outputs, labels, topk=(1,))
                val_loss.update(loss.item(), inputs.size())
                val_acc.update(acc1[0], inputs.size())

        # record the values in tensorboard
        writer.add_scalar('loss/val', val_loss.avg , epoch + 1)  # average loss
        writer.add_scalar('acc/val', val_acc.avg , epoch + 1)  # average acc
        
        # ===== save the model =====
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'alexnet',
                'val_loss' : val_loss.avg,
                'val_acc': val_acc.avg,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()},
                MODEL_PATH, epoch + 1)

    print('Finished Training')
    print("Training time elapsed: {:.4f}mins".format((time.time() - train_time) / 60))
    print()
    
    writer.close()  # close tensorboardX writer
    

if __name__ == '__main__':
    run_time = time.time()
    main()
    mins = (time.time() - run_time) / 60
    hours = mins / 60
    print("Total run time: {:.4f}mins, {:.4f}hours".format(mins, hours)) 

