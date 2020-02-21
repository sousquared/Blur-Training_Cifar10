# BlurNet-Cifar10

## Architecture
**AlexNet-Cifar10**  
This AlexNet has different kernel-size and dense-size due to the image size of Cifar10. This AlexNet is the same structure with [this site (in Japanese)][1].


## Training Scripts
In all training scripts, you need to use `--exp-name` or `-n` option to define your experiment's name. Then the experiment's name is used for managing results under `logs/` directory.   

- `train_normal.py`   
train Normal alexnetCifar10

- `train_blurall.py`  
blur ALL images in the training mode.  

- `train_blurhalf.py`  
blur first half epochs (e.g. 30 epochs) in the training mode.
```bash
python train_blurhalf.py -s 1 -k 7 7 -n blurhalf_s1_k7-7
```
- `train_blurstep.py`  
blur images step by step (e.g. every 10 epochs).  


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