 # Deploying  model to web.
 #### Front page
 ![alt text](https://github.com/rakshitrk/dog_breed/blob/master/images/index.jpg)
 #### Shows predicted breed and information about breed
 ![alt text](https://github.com/rakshitrk/dog_breed/blob/master/images/result.jpg)
 
 ## About
 > This is example of transfer learning. Deployed using Flask frontend  is designed   using css, javascript .
 vgg19(pretrained on imagenet) to find wether image contains dog or not and resnet152 to find the breed of dog.
 dataset of 133 dog breeds is used to train the model 
>> [Dog breed Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- models folder contains  resent and vgg model weight files instead of read.txt
 
 > Installation & usage :
 - Download or clone the repository:
   >> git clone https://github.com/spctr01/dog_breed.git
   
 - move into folder:
   >> cd dog_breed
   
 - Install the requirements:
   >> pip install requirements.txt
   
 - run the commands(running flask app):
   >>  export FLASK_APP=app.py
   >> flask run
       
   

 
 
 
 
 
