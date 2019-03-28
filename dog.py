
import torch
import torchvision.models as models
import numpy
from PIL import Image
import torchvision.transforms as transforms
from train import loaded_model
import pretrainedmodels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



#transforms and return image
def load_image(path):    
    image = Image.open(path).convert('RGB')
    
    transform = transforms.Compose([
                        transforms.Resize(size=(244, 244)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]) 

    image = Image.open(io.BytesIO(path))
    image = transform(image)[:3,:,:].unsqueeze(0)
    return image


#loading pretrained Xception model
model_name = 'xception'
model = pretrainedmodels.__dict__[model_name](num_classes = 1000, pretrained = 'imagenet')
model = model.to(device)


#returns  true  false respective to dog detected or not
def predict(path):
  '''
    vgg19 is trained on  imagenet containg 1000 classes 
    so from class no. 151 to 277 reprsents the dogs
  '''
  img = load_image(path)
  img = img.to(device)
  ret = model(img)
  class_no = torch.max(ret,1)[1].item()
  returns (class_no >=151 and class_no <= 277)  


# pass the image to trained model and predict the breed.
def breed(path):
  in_image = predict(path)
  

def wiki(info):
  return wikipedia.summary(info)



