# Fine-tuning inceptionV2 in Tensorflow

## In this repository you will find the code to fine-tune an inceptionV2 model (pre-trained on Imagenet) in Tensorflow.

In particular I applied it to the CUB-200-2011 Birds Dataset, which is a famous dataset for fine-grained classification that contains ~6k training and ~6k test images of 200 species of birds. If you want to apply it to another dataset you can do it and I'll add some details later in order to show you how. 

I decided to train the inceptionV2 model because it is rather small and light but has good performances. Also it uses batch normalization which let us improve the training speed drastically by using an high learning rate (0.1). 

#### Without bounding-boxes annotations, the network reaches ~79% accuracy with a 224x224 input. 

The structure of the code has been taken from this great gist: https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c and modified to train an InceptionV2 model. Some details have been taken from this great repository of Visipedia which I recommend: https://github.com/visipedia/tf_classification/wiki/CUB-200-Image-Classification. The training details and parameters have been taken from the inceptionV2 paper https://arxiv.org/abs/1512.00567 in which there is specific section related to fine-grained classification.

I wanted to make the code as easy and as low-level as possible, in order to be able to understand all the things that are going on. If you are trying to learn like me, I think it's key to stay a little bit lower level to get a better feel of what is really happening. This code does not use the slim.train definition for this reason, all the relevant bits of the slim.train function has been taken so nothing should be hidden. 

### Structure of the code

The repository is made by two files which contain the network definition and the training process definition. There is no hidden code anywhere, all that runs is in these two files. 

1. **inceptionV2_net.py:** contains the definition of the network and nothing else. I deleted all the things that were not necessary to make this code run. I did that because I found the inceptionV2 structure quite complex to understand as it was written in the slim repository, so now it should be easier to understand. If you want to take a look at the original code by slim is here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v2.py 

2. **inceptionV2_train.py:** contains the definition of the graph which is needed to train the model on the dataset. As you can see, it uses the new Dataset API which is really easy to use and also quite fast. 


## In particular this code:

- **Loads the pre-trained inceptionV2 model from ckpt**

- **Loads the dataset from files:** it expects jpg images placed in a pre-defined structure. The structure of the dataset must be the following: a single main folder containing all the dataset. In this main folder there is a single folder for each class. Each sub-folder has the name of the class. In each sub-folder there are the images corresponding to that class. This dataset organization is already done if you are using the CUB-200 dataset. 

- **Preprocesses the data:** once the files are loaded, they are preprocessed as the inceptionV2 wants. All images will be isotropically rescaled to have smallest side = 256 and put in range [-1,+1] and then training images will be randomly cropped and flipped while test images will be centrally cropped. There is also the possibility to add color augmentation with the 'color_augm_probability' flag which will augment the training data given a probability in range [0,1].

- **Trains the model:** after preprocessing the data is fed into the model, one batch at a time. You can define the log frequency of the training process in the args. The optimizer is SGD with momentum. You can also use label smoothing and gradient clipping if you want to experiment a bit. 

- **Evaluates the model:** during training there is the possibility to compute the accuracy of the current model on the test set. You can decide at which epoch you want to start doing it, at which frequency and how many batches to check every time. 

- **Saves the model:** the best model is saved every time the best test accuracy is increased. The best accuracy starts at 0.0 and hopefully gets better. 

### Requirements

I used Tensorflow 1.2, Numpy, Python 2.7. I run my tests on a NVIDIA K40 GPU which can handle a batch size of 256, if you do not have enough memory you will need to change the batch size to 128,64,32 or even lower. 

### If you want to use this code

- **Download the inceptionV2 ckpt** from http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz and untar it. Then modity the 'ckpt_dir' in the args.
- **Download the CUB-200-2001 Dataset** from the official source: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz and untar it where you want. 
- **Now you need to split the dataset into train and test part:** change the paths into split.py accordingly to the paths that you have in your system and then simply launch it:

       python split.py
    
It should create two separate folders "train" and "test" in the paths that you declared in the file. NOTE: split.py deletes the greyscale images and saves only the RGB images. This is done because the current implementation of the model does not accept greyscale images. The total number of RGB images is 5990 train images, 5790 test images.
- **Check the hyperparameters** such as the learning rate. Now the code only works with a decay rate defined with the 'lr_decay_rate'. It decreases the lr every 'decrease_lr_every_epoch' epochs. My current best model starts at 0.1 and decreases it at 0.01 after 10k iterations and at 0.001 after 20k iterations. 
- **Check whether you want to change some of the logging frequencies** in the args. Double check the once referring to the validation accuracy computation such as 'val_acc_every_n_epochs' and 'batches_to_check' which define when you want to compute the validation accuracy and how many batches of test data you want to check every time. Be careful that if you keep the current configurations, the test accuracy will be computed each epoch on all the test set! On the CUB-200 test set with an NVIDIA K40 it takes ~30secs.
- **Change the paths** to the dataset in the code or by the args and launch

      python inceptionV2_train.py

### What should happen

If you have done everything correctly, the learning should start and can be monitored via tensorboard by typing into another terminal:

      tensorboard --logdir=/path/to/log/dir/ 
      
The output should be something like this:

![alt text](https://github.com/simo23/inceptionV2_finetune/blob/master/training_tensorboard.png "tensorboard")

You can notice **severe overfitting** due to the limited amount of training data. The training loss does not go to zero because of the label smoothing and l2 loss. You can see that the network has converged and is stable after ~30k iterations so I stopped the training. 

       
       


