# bike
machine learning on bike sharing data
data and description from https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset


## Setup

```sh
git clone https://github.com/mar-volk/bike
```

```sh
conda env create --file environment.yml
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

copy the raw data from above mentioned website (i.e. day.csv, hour.csv) to data/raw/.
## Usage

```sh
conda env update --file environment.yml
source activate bike
jupyter notebook
```
 Run "src/prep/train_val_test_split.py" to prepare splitted, production-like data.
 
 