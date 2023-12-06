# Adversary-Black-Box-Attacker
final project for ECE 685 - Duke University


## Setting up Environment and Directory Structure 

Set up a conda environment and download the following packages indicated in the `requirements.txt` file. 
```
conda create -n "myEnv" python=3.10 
conda activate myEnv 
```
Clone this repo and make `data` directory. Then, download the datasets with the following commands, unzip them, and move them to this directory. Note that you must have kaggle set up on your machine to download it from the CLI. Otherwise, you can download the dataset [here](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species?rvi=1). 
```
wget https://pjreddie.com/media/files/imagenet64.tar
kaggle datasets download -d gpiosenka/butterfly-images40-species
```
The tree structure should look like this after completion. 
```
├── data
│   ├── butterflies_and_moths
│   │   ├── butterflies and moths.csv
│   │   ├── test
│   │   ├── train
│   │   ├── training.csv.csv
│   │   └── valid
│   └── imagenet64
│       ├── train
│       └── valid
```

