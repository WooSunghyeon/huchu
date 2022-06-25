import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from mobilenetv2 import MobileNetV2, ReinvertedNet
from utils import progress_bar
import numpy as np

#%%
parser = argparse.ArgumentParser(description='Auxiliary Activation Learning')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', type=int, 
                    default=512, help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, 
                    default=512, help='input batch size for teset (default: 512)')
parser.add_argument('--epochs', type=int, default=90,  
                    help='number of epochs to train (default: 90)')
parser.add_argument('--dataset', type= str ,choices = ['mnist', 'cifar10', 'cifar100', 'svhn', 'tiny-imagenet', 'imagenet'], 
                    default='cifar10', help='choose dataset (default: cifar10)')
parser.add_argument('--model', type= str ,choices = ['mobilenetv2', 'reinvertednet'], 
                    default='mobilenetv2', help='choose architecture (default: mobilenetv2)')
parser.add_argument('--optimizer', type=str, choices = ['sgd', 'adam'], default = 'sgd'
                    , help = 'choose optimizer (default : sgd)')
parser.add_argument('--scheduler', type=str, 
                    choices = ['step', 'cos'],
                    default='step',  help='scheduler to use (default: step)')
parser.add_argument('--device', nargs='+', type= int,  help='device_num')
parser.add_argument('--checkpoint', action = 'store_true', default = False,
                    help = 'use checkpoint')

def main():
    args = parser.parse_args()
    device = args.device
    if type(device) != int:
        device = args.device[0]
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    # All data licensed under CC-BY-SA.
    dataset = args.dataset
    if dataset == 'mnist':
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.test_batch_size, shuffle=False)
    
    elif dataset == 'svhn':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'train',download=True, transform=transform),
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'test',download=True, transform=transform),
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False)
        
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
                
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
        
    elif dataset == 'tiny-imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        num_workers = 16
        trainset = torchvision.datasets.ImageFolder('/nfs/home/wshey/tiny-imagenet/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder('/nfs/home/wshey/tiny-imagenet/val', transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=num_workers)
        
    elif dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        num_workers = 4 * len(args.device)
        trainset = torchvision.datasets.ImageFolder('/scratchpad/datasets/ImageNet/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder('/scratchpad/datasets/ImageNet/val', transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=num_workers)
        
    # Set the model               
    net = MobileNetV2()
    if args.model == 'reinvertednet':
        net = ReinvertedNet()
    save_file_name = args.dataset + '_' + args.model 
    
    if args.checkpoint:
        checkpoint = torch.load('./checkpoint/' + save_file_name + '.pth')
        state_dict = checkpoint['net']
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['acc']
        net.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
            
    if type(args.device) != int:
        net = nn.DataParallel(net, device_ids=args.device) 
        net = net.to(device)
    else:
        net = net.to(device)
    
    print(net)
    print('net memory: ' + str((torch.cuda.memory_allocated(args.device[0])) / 1024 /1024) + 'Mib')

    # Decide optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #for 
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=0)
    
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.checkpoint:
        scheduler.last_epoch = start_epoch - 2
    
    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = correct = total= step=max_mem= 0
        torch.cuda.empty_cache()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            pre_allocated = torch.cuda.memory_allocated(args.device[0]) / 1024 /1024
            outputs = net(inputs)
            post_allocated = torch.cuda.memory_allocated(args.device[0]) / 1024 /1024  
            act_mem = (post_allocated-pre_allocated)*len(args.device)
            #print(act_mem)
            loss = criterion(outputs, targets) 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            step += 1
            
            if act_mem>max_mem and step > 10:
                max_mem=act_mem
               
        return 100.*correct/total, train_loss, max_mem
        
    def test(epoch, best_acc):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        acc = 100.*correct/total

        return acc, test_loss
    
    result = []
    max_mem = 0
    
    if args.checkpoint:
        result = np.load("results/" + save_file_name + '.npy')[:start_epoch,:].tolist()
        
    for epoch in range(start_epoch, args.epochs):
        train_acc, train_loss, act_mem = train(epoch)
        test_acc, test_loss = test(epoch, best_acc)
        scheduler.step() 
        if not os.path.isdir('results'):
            os.mkdir('results')
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + save_file_name + '.pth')
            best_acc = test_acc
            print('best accuracy : ' + str(round(best_acc, 2)), 'epoch : ' + str(epoch) + '\n')
        if act_mem > max_mem:
            max_mem = act_mem
        print('training_memoy: ' + str(round(max_mem, 2)) + 'Mib', 'epoch : ' + str(epoch) + '\n')
        
        result.append([train_acc, test_acc, train_loss, test_loss])
        np.save("results/" + save_file_name, result)
        
if __name__ == '__main__':
    main()      
