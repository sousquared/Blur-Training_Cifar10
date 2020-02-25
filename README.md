# BlurNet-Cifar10

## Preparation
Install Python Packages  
```bash
$ pip install -r requirements.txt
```
Or pull and run [docker image][4] (e.g. blurnet:1.0) I made for this experiments.  

## Architecture
**AlexNet-Cifar10**  
This AlexNet has different kernel-size and dense-size due to the image size of Cifar10. This AlexNet is the same structure with [this site (in Japanese)][1].


## Training Scripts
In all training scripts, you need to use `--exp-name` or `-n` option to define your experiment's name. Then the experiment's name is used for managing results under `logs/` directory.   

- `train_normal.py`   
train Normal alexnetCifar10.  
usage example:  
```bash
$ python train_normal.py -n normal_60e
```
- `train_blur-all.py`  
blur ALL images in the training mode.  
usage exmaple:  
```bash
$ python train_blur-all.py -s 1 -k 7 7 -n blur-all_s1_k7-7
```

- `train_blur-half.py`  
blur first half epochs (e.g. 30 epochs) in the training mode.
usage example:  
```bash
$ python train_blur-half.py -s 1 -k 7 7 -n blur-half_s1_k7-7
```
- `train_blur-step.py`  
blur images step by step (e.g. every 10 epochs).  
usage example:  
```bash
$ python train_blur-step.py -n blur-step
```


## logs/

`logs/` directory will automaticaly be created when you run one of training scripts.  
`logs/` directory contains `outputs/`, `models/`, and `tb/` directories.  

- `logs/outputs/` : records "stdout" and "stderr" from the training scripts.
- `logs/models/` : records model parameters in the form of pytorch state (default: every 10 epochs). 
- `logs/tb/` : records tensorboard outputs. (acc/train, acc/val, loss/train, loss/val)

## data/ : Cifar10
`data/` directory will automaticaly be created when you run one of training scripts.  


## citation
Training scripts and functions are strongly rely on [pytorch tutorial][2] and [pytorch imagenet trainning example][3].

[1]:http://cedro3.com/ai/pytorch-alexnet/
[2]:https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
[3]:https://github.com/pytorch/examples/blob/master/imagenet/main.py
[4]:https://hub.docker.com/r/sousquared/blurnet