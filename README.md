# Adversary-Black-Box-Attacker
final project for ECE 685 - Duke University


## Setting up Environment and Directory Structure 

Create a new conda environment with the following packages indicated in the `environment.yml` file. 
```
conda env create -f environment.yml 
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
To run NES, go back into the main directory and run 
```
python eval_nes.py --lr 0.01 --target_eps 0.05 --n_samples 100 --sigma 0.001 --dataset_size 100 --dataset imagenet64 --max_queries 20000
```

To run HopSkipJumpAttack, change the params variable in `eval_hsj.py` to adjust the dataset, defender, and parameters. Then, run
```
python eval_hsj.py device_id data_modulo
```

where `data_modulo` affects which slice of the data is used. 