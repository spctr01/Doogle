import os
import io
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy
from PIL import Image
import wikipedia




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



#transforms and return image
def load_image(path):    
    
    
    transform = transforms.Compose([
                        transforms.Resize(size=(244, 244)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]) 

    image = Image.open(path)
    image = transform(image)[:3,:,:].unsqueeze(0)
    return image


#loading pretrained  vgg19 model
model = models.vgg19(pretrained = False)
model.load_state_dict(torch.load('Models/vgg.pth', map_location = device))
model.to(device)
model.eval()

#loading resnet101 trained on dataset
#finetune model 
resnet= models.resnet152(pretrained = False)
ftrs = resnet.fc.in_features     # gives input dimentions of fullyconnected layer
resnet.fc = nn.Linear(ftrs,133)
resnet.load_state_dict(torch.load('Models/model.pt', map_location = device))
resnet.to(device)


#returns   dog detected or not
def vgg(path):

  '''
      vgg19 is trained on  imagenet containg 1000 classes 
     so from class no. 151 to 277 reprsents the dogs(including wild)
  '''
  output= model(path)
  return torch.max(output, 1)[1].item()


#returns predicted breed
def res(path):
  output = resnet(path)
  return torch.max(output,1)[1].item()
	

#reading class_name if not a dog from vgg classes
def class_name_vgg(idx):
  file = open('classes/vgg.txt', 'r')
  lines = file.read().split('\n')
  lines = [x for x in lines]
  return lines[idx]

#returns breed name from text file
def breed_name(idx):
  file = open('classes/breed.txt', 'r')
  lines = file.read().split('\n')
  lines = [x for x in lines]
  return lines[idx]



# pass the image to trained model and predict the breed.
def breed(path):
  in_img = load_image(path)
  a = vgg(in_img)

  if a >= 151 and a <=280:
    class_no = res(in_img)
    found_breed = breed_name(class_no)
    flag = 0
    return found_breed, flag
  else:
    found_obj = class_name_vgg(a) #returns class  from vgg to show what is in image
    flag = 1
    return found_obj, flag

#returns information from wikipedia
def wiki(info):
  return wikipedia.summary(info)

