 # Deploying  model to web.
 #### Front page
 ![alt text](https://github.com/rakshitrk/dog_breed/blob/master/images/index.jpg)
 #### Shows predicted breed and information about breed
 ![alt text](https://github.com/rakshitrk/dog_breed/blob/master/images/result.jpg)
 
 ## About
 >  Frontend  is designed in flask using css, javascript vgg19(pretrained on imagenet) to find weather image contains dog or not and resnet152 to find the breed of dog.
 dataset of 133 dog breeds is used to train the model (as per the model  the total images in dataset are not sufficient any other
 dog breed dataset can be used)
>> [Dog breed Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- models folder contains  resent and vgg model weight files instead of read.txt
 
 ## Requirements
 - PyTorch1.0 
 - Flask-1.0.2 
 - Wikipedia api 
 
 
 
 
