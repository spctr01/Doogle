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
 
    
## Installation & usage

Download or clone the repository by :
```sh
git clone https://github.com/spctr01/dog_breed.git
```
move into folder:
```sh
 cd dog_breed
 ```

Install the requirements:
```sh
pip install -r requirements.txt
```

Download the model weight files and 
paste the files (vgg.pth & model.pt) to Models Folder 

[`Resnet152`](https://www.kaggle.com/rakshitrk/resnet152-dog-breed?) 

[`Vgg19`](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) (change name to vgg.pth)

run the commands(running flask app):
```sh
export FLASK_APP=app.py
flask run
```

       
   

 
 
 
 
 
