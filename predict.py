##
#Imports libraries to create pretrainde model and normalize images
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
    description='This program predict image class by trained network',
)

parser.add_argument('image_path', action="store",type=Path, default=False)
parser.add_argument('checkpoint_directory', action="store", type=str, default=False)
parser.add_argument('--top_k', action="store", dest='top_k',type=int, default=5)
parser.add_argument('--category_names', action="store", dest='category_names', type=str, default="flowers_name.json")
parser.add_argument('--gpu', action="store_true", dest='gpu', default=False)

args = parser.parse_args()

device=torch.device('cpu')

#Load checkpoint and get GPU device if possible
if args.gpu & torch.cuda.is_available():
    device = torch.device("cuda")
    print('Choosen device is {}'.format(device))    
    
    if args.checkpoint_directory=="":
        checkpoint = torch.load('checkpoint.pth')
    else:
        checkpoint = torch.load(args.checkpoint_directory+'/checkpoint.pth')

else:
    print('Choosen device is {}'.format(device))

    if args.checkpoint_directory=="":
        checkpoint = torch.load('checkpoint.pth', map_location='cpu')
    else:
        checkpoint = torch.load(args.checkpoint_directory+'/checkpoint.pth', map_location='cpu')

print('The checkpoint is loaded')             

#Create pretrained model
pretrained_model=models.__dict__[checkpoint['arch']](pretrained=True)

if "resnet" in checkpoint['arch']:
  pretrained_model.fc=checkpoint['fc']
  pretrained_model.hidden_layers=checkpoint['hidden_layers']
else:
  pretrained_model.classifier=checkpoint['classifier']
  pretrained_model.features=checkpoint['features']
    
pretrained_model.class_to_idx=checkpoint['class_to_idx']
pretrained_model.load_state_dict=checkpoint['state_dict']    

#for param in pretrained_model.parameters():
#   param.requires_grad = False

criterion=nn.NLLLoss()

optimizer = optim.SGD(pretrained_model.parameters(), lr=checkpoint['learning_rate'])
optimizer.state_dict=checkpoint['optimizer_state_dict']
pretrained_model.to(device)

with active_session():
    
    #Process an image before prediction
    mean = np.array([485, 456, 406])
    std = np.array([229, 224, 225])
    
    pil_image = Image.open(args.image_path)
    
    #Image scaling
    old_aspect = float(pil_image.width)/float(pil_image.height)
    width=256
    height=int(width*pil_image.height/pil_image.width)
    new_aspect = float(width)/float(height)

    if old_aspect < new_aspect:
        height = int(width / old_aspect)
    else:
        width = int(height * old_aspect)
        
    #Crop image to 224x224
    pil_image=pil_image.resize((width,height))
    pil_image=pil_image.crop((0,0,224,224))
    
    #Normalize image
    np_image = np.array(pil_image)    
    np_image = (np_image-mean)/std  
    np_image=np_image.transpose((1,2,0))
    np_image=np_image.astype(np.float32)
    

    # Get prediction
    pretrained_model.eval()
    
    if args.gpu & torch.cuda.is_available(): 
        image=torch.from_numpy(np_image)
        ps=torch.exp(pretrained_model.forward(image.cuda().view(1,3,224,224)))
        top_p, top_class = ps.topk(args.top_k, dim=1)  
    else:  
        image=torch.from_numpy(np_image)
        ps=torch.exp(pretrained_model.forward(image.view(1,3,224,224)))
        top_p, top_class = ps.topk(args.top_k, dim=1)    
    
    #Print results with class labels
    import json
    with open(args.category_names, 'r') as f:
        categories = json.load(f)
    
    labels=[categories[str(item)] for item in top_class.view(-1).tolist()]
    ps=top_p.view(-1).tolist()
    print('Results:')

    for i in range(len(ps)):
          print("{} - {:.3} %".format(labels[i], ps[i]*100))
            