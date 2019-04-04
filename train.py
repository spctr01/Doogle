

import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



#transforms the images of train, validation, test datasets
transform = {'train':transforms.Compose([transforms.Resize((244,244)),
                                      
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]),
             
             'valid':transforms.Compose([transforms.Resize((244,244)),
                                     
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]),
             
             'test':transforms.Compose([transforms.Resize((244,244)),
                                  
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
}




bs = 10
num_epochs = 15
#pytorch's ImageFolder to load the data
data = 'dogImages/'

train_data = datasets.ImageFolder(os.path.join(data, 'train/'), transform = transform['train']) 
val_data =  datasets.ImageFolder(os.path.join(data, 'train/'), transform = transform['valid']) 
test_data = datasets.ImageFolder(os.path.join(data, 'train/'), transform = transform['test']) 

train_loader = torch.utils.data.DataLoader(train_data, batch_size = bs, num_workers = 2, shuffle = True) 
val_loader = torch.utils.data.DataLoader(val_data, batch_size = bs, num_workers = 2, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, num_workers = 2, shuffle = False)




#finetuning pretrained resnet101 model 
model = models.resnet101(pretrained = True)

ftrs = model.fc.in_features     # gives input dimentions of fullyconnected layer
model.fc = nn.Linear(ftrs,133)  # redesigning fully connected layer with 133 nodes(dog breeds)

model = model.to(device)
print(model)
  
  

#loss and optimizer for model
critersion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.002)




val_loss_min = np.Inf  # assigning minmum validation loss to infinity
  
for epoch in range(num_epochs):
    
  train_loss = 0.0  #running training loss
  val_loss = 0.0    #running validation loss
     
    
  #training model
  model.train()     #preparing model to train
  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
      
    optimizer.zero_grad()
    output = model(inputs)
    loss = critersion(output, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*inputs.size(0)
      
  #evalutaing model
  model.eval()     #preparing model for validation
  for inputs, labels in val_loader:
    inputs =  inputs.to(device)
    labels = labels.to(device)
    
    output = model( inputs)
    loss = critersion(output, labels)
    val_loss += loss.item()* inputs.size(0)
        
  train_loss = train_loss/len(train_loader.dataset)
  val_loss = val_loss/len(val_loader.dataset)
      
  print('Epoch:{}\nTraining Loss:{:6f}\tValidation Loss:{:6f}'.format(epoch,train_loss,val_loss))
      
  if val_loss <= val_loss_min:
    print('Validation Loss Dec. {:6f}--->{:6f}\t...SAVING...'.format(val_loss_min,val_loss))
    torch.save(model.state_dict(),'Models/model.pt')         #saves model when validation loss dec. from min. validation loss
    val_loss_min = val_loss
      
  



#loading model
loaded_model = model.load_state_dict(torch.load('Models/model.pt'))

#testin trained model
correct = 0 # total no of correct predictions
total = 0   # total no of predictions / data

model.eval()  #preparing model for testing
for data in test_loader:
  
  images,labels = data
  images = images.to(device)
  labels = labels.to(device)
  
  output = model(images)
  _, pred = torch.max(output,1)
  
  total += labels.size(0)
  correct += (pred == labels).sum().item()
  
  acc = 100*correct/total
print('Accuracy of model:{} %'.format(acc))

