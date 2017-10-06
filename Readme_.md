
# How to generate the submission file

## Dependencies

Language: Python 3.6.1

I installed Anaconda 4.4.0 (64-bit), and list of libraries I used is below.
	
|library|version|
|:-:|:-:|
|tensorflow|1.3.0|
|cv2	|3.3.0|
|pandas	|0.20.1|
|numpy	|1.12.1|
|keras	|2.0.6|
|sklearn		|0.18.1|
|tqdm	|4.15.0|

I mainly used windows 10 with a GPU GTX1080-Ti. My scripts need at least 11GB GPU memory.

## Directories & Datas

* 'Carvana-All-Files' is root directory.
* (*) means it can be downloaded from competition site, and I didn't included their content in solution file.
* 'train0~12.py' means train0.py, train1.py, ... , train12.py.

```
Carvana-All-Files (root)
├── csv_log
├── input
|   ├── train_hq (*)
|   ├── test_hq (*)
|   ├── train_masks
│   └── corrected_masks
├── logs
├── model
|   ├── losses.py
|   └── mdcb.py  
├── prob_maps
|   ├── fold1~12_lr20_1024 (fold10 & fold2~tta are not included)
|   ├── fold4fold2_ensemble
|   ├── fold2456_ensemble
|   ├── fold12345_ensemble
|   ├── fold12345689_ensemble
|   ├── full_ensemble
|   └── test_masks
├── submit
├── weights
|   └── fold1~12_lr20_1024_model.hdf5
├── Readme.md
├── Carvana-3rd-Place-Solution-Report.pdf
├── train0~12.py (train10.py is not included)
├── predict0~12.py (predict10.py is not included)
├── ensemble1~5.py
├── pretraining1~2.py
├── image_process.py
├── run.sh
└── run_no_train.sh
```

## Generate submission file

If you would like to generate submission file by using trained models, it is OK that you just run './run_no_train.sh'.

If you would like to generate submission file from scratch(including train phase), you need to do some processes. 

First, you need to use mask images in train_masks that provided in my solution zip file.
Location of train_masks folder is 'Carvana-All-Files/input/train_masks'.

For your reference, generating process of my images in train_masks/ is shown below.

1. Convert mask files format *.gif to *.png .
2. Replace the 31 masks(in train_masks/) 
with './input/corrected_masks' which provided in https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37229 by JandJ.
(Note that I also converted it to png.)

Then, run './run.sh'. It may take so much time.
