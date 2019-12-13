# flower-image-classifier
This repository contains the console example of training **a convolutional neural network** and recognize different species of flowers for choosen image. A neural network will be train on [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) with 102 categories. The program was created during [Udacity's AI Programming with Python Nanodegree Program](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).

## Dependencies
The next points are necessary for correct execution:
- **Python 3.6** or later
- **Torch 0.4.0**
- **Pillow 5.2.0**
- **torchvision 0.2.1**

Please, check your environment.
Also this repository contains `workspace_utils.py`, which is important for long-running work,  for example, training a network. So you need install `requests` package on an environment.

## Quickstart
To train and use pretrained model you need run `train.py` and `predict.py` scripts.
Firstly, `train.py` has the one required argument - `data_dir`, which takes in the path to a directory with training images. The repository contains directory `flowers` already, so you can train a network this way:
```
python train.py flowers/
```
**By defaulf** script creates a **VGG19** model with **512** hidden units, trains it on **10** epochs with learning rate - **0.01**, dropout - **0.2** in **CPU** mode (it could takes a lot of time) and saves the trained model to `checkpoints/checkpoint.pth`. The checkpoint could be a quite large file. Usually 130 Mb or more.

After that you can test this trained neural network with a flower image. `predict.py` need the path to a choosen image and the path to checkpoint as required arguments.
```
python predict.py flowers/valid/3/image_06631.jpg checkpoints/
```
**By default** script uses `flowers_name.json` as **flower labels**, prints top **5** predictions and runs in **CPU** mode (it could takes a lot of time). When the scripts is completed you'll see results like this:
---image----
## Traning a neural network
`train.py` takes in the next optional arguments:
- `--save_directory` - the directory for saving checkpoints (default=`checkpoints/`)
- `--arch` - the architecture of a model (default=VGG19). The full list you can see [here](https://pytorch.org/docs/stable/torchvision/models.html)
- `--learning_rate` - the learning rate for training (default=0.01). Try to find the most suitable one.
- `--hidden_units` - the number of hidden units in a model (default=512). More is better.
- `--epochs` - the number of training epochs (default=10). In general 30 is OK to train a network completely.
- `--dropout` - the value of dropout during training (default=0.2). The default value is common.
- `--gpu` - The running mode (default=CPU). Training could takes a lot of time, so GPU is the great idea (if your hardware supports GPU). There is the "how to check"  instruction [here](https://developer.nvidia.com/cuda-gpus#compute). 
## Using a pretrained model for prediction
`predict.py` takes in 3 optional arguments:
- `--top_k` - The number of top K predictions to print as results (default=5).
- `--category_names` - A `json` file with species of flowers (default=`flowers_name.json`). 
- `--gpu`- The running mode (default=CPU). The prediction is quite short process, so GPU isn't nesessary.

## Issues
- The scripts work with `JPEG` images

## License
The contents of this repository are covered under the [MIT License](https://github.com/UlianaDzhumok/flower-image-classifier/blob/master/license.txt).