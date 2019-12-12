##
#Imports libriaries to work with CNN model and calculations
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np

import argparse
from pathlib import Path

from workspace_utils import active_session


#Parse input parameters
parser = argparse.ArgumentParser(
    description='This program trains a new neural network with a defined architecture and hyperparameters',
)

parser.add_argument('data_dir', action="store",type=Path, default=False)
parser.add_argument('--save_directory', action="store", dest='save_directory', type=str, default="")
parser.add_argument('--arch', action="store", dest='arch',type=str, default='vgg19')
parser.add_argument('--learning_rate', action="store", dest='learning_rate', type=float, default=0.003)
parser.add_argument('--hidden_units', action="store", dest='hidden_units', type=int, default=512)
parser.add_argument('--epochs', action="store", dest='epochs', type=int, default=10)
parser.add_argument('--dropout', action="store", dest='dropout', type=float, default=0.2)
parser.add_argument('--gpu', action="store_true", dest='gpu', default=False)

args = parser.parse_args()

device=torch.device("cpu")
steps=0
print_every=10

#Create transform and datasets load and processing images
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                         [0.229,0.224,0.225])])

image_dataset = datasets.ImageFolder(args.data_dir,transform=data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset,batch_size=32,shuffle=True)


#Create a new model with defined architecture and custom classifier
model=models.__dict__[args.arch](pretrained=True)

#for param in model.parameters():
#    param.requires_grad = False

from collections import OrderedDict
classifier=nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(25088,args.hidden_units)),
                                ('relu',nn.ReLU()),
                                ('dropout',nn.Dropout(p=args.dropout)),                           
                                ('fc2',nn.Linear(args.hidden_units,102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

if "resnet" in args.arch:
  model.fc=classifier
else:
  model.classifier=classifier

criterion=nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)


#Get GPU device if possible
if args.gpu & torch.cuda.is_available():
    device = torch.device('cuda')
print('Choosen device is {}'.format(device))

model.to(device)

#Train our new newtwork
with active_session():
    for e in range(args.epochs):

        running_loss=0

        if args.gpu & torch.cuda.is_available():

            for images,labels in dataloader:
                images,labels=images.to(device),labels.to(device)

                #Clear gradients
                optimizer.zero_grad()
                steps+=1

                logps=model(images)
                loss=criterion(logps,labels)
                loss.backward()
                optimizer.step()

                running_loss+=loss.item()

                #Printing training and validation loss to check overfitting
                if steps % print_every==0:

                    test_loss=0
                    accuracy=0
                    model.eval()         

                    with torch.no_grad(): 

                        for images, labels in dataloader:

                            logps = model(images.cuda())
                            test_loss+=criterion(logps, labels.cuda())
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels.cuda().view(*top_class.shape)
                            accuracy+= torch.mean(equality.type(torch.cuda.FloatTensor))

                    print('Epoch {}/{}.. '.format(e+1, args.epochs),
                          'Training loss: {:.3f}'.format(running_loss/len(dataloader)),
                          'Validation loss: {:.3f}'.format(test_loss/len(dataloader)),
                          'Accuracy: {:.3f}'.format(accuracy/len(dataloader))) 

                    model.train()
        else:

            for images,labels in dataloader:
                images,labels=images,labels

                #Clear gradients
                optimizer.zero_grad()
                steps+=1

                logps=model(images)
                loss=criterion(logps,labels)
                loss.backward()
                optimizer.step()

                running_loss+=loss.item()

                #Printing training and validation loss to check overfitting
                if steps % print_every==0:

                    test_loss=0
                    accuracy=0
                    model.eval()         

                    with torch.no_grad(): 

                        for images, labels in dataloader:

                            logps = model(images)
                            test_loss+=criterion(logps, labels)
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy+= torch.mean(equality.type(torch.FloatTensor))

                    print('Epoch {}/{}.. '.format(e+1, args.epochs),
                          'Training loss: {:.3f}'.format(running_loss/len(dataloader)),
                          'Validation loss: {:.3f}'.format(test_loss/len(dataloader)),
                          'Accuracy: {:.3f}'.format(accuracy/len(dataloader))) 

                    model.train()

#Save network checkpoint after training for prediction later
if "resnet" in args.arch:
  checkpoint = {'arch':args.arch,
              'fc': model.fc,
              'class_to_idx': image_dataset.class_to_idx,
              'optimizer_state_dict':optimizer.state_dict(),
              'hidden_layers': model.hidden_layers,
              'learning_rate':args.learning_rate,
              'state_dict': model.state_dict()}
else:
  checkpoint = {'arch':args.arch,
              'classifier': classifier,
              'class_to_idx': image_dataset.class_to_idx,
              'optimizer_state_dict':optimizer.state_dict(),
              'features': model.features,
              'learning_rate':args.learning_rate,
              'state_dict': model.state_dict()}

if args.save_directory=="":
  torch.save(checkpoint, args.save_directory+'checkpoint.pth')
else:
  torch.save(checkpoint, args.save_directory+'/checkpoint.pth')
