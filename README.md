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


## Usage
In all training scripts, you need to use `--exp-name` or `-n` option to define your experiment's name. Then the experiment's name is used for managing results under `logs/` directory.   
You can choose the training mode from {normal,blur-all,blur-half,blur-step,blur-half-data} by using `--mode [TRAINING MODE]` option.

- **normal**  
train Normal alexnetCifar10.  
usage example:  
```bash
$ python main.py --mode normal -e 60 -n normal_60e
```

- **blur-all**  
blur ALL images in the training mode.  
usage exmaple:  
```bash
$ python main.py --mode blur-all -s 1 -k 7 7 -n blur-all_s1_k7-7
```

- **blur-half**    
blur first half epochs (e.g. first 30 epochs in 60 entire epochs) in the training.
usage example:  
```bash
$ python main.py --mode blur-half -s 1 -k 7 7 -n blur-half_s1_k7-7
```

- **blur-step**  
blur images step by step (e.g. every 10 epochs).  
usage example:  
```bash
$ python main.py --mode blur-step -n blur-step
```

- **blur-half-data**    
blur half training data.
usage example:  
```bash
$ python main.py --mode blur-half-data -s 1 -k 7 7 -n blur-half-data_s1_k7-7
```

- `--resume [PATH TO SAVED MODEL]`   
train Normal alexnetCifar10 from your saved model.  
usage example:  
```bash
python main.py -e 90 --mode normal --resume ../logs/models/blur-half_s1_k7-7/model_060.pth.tar -n blur-half_s1_k7-7_from60e
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