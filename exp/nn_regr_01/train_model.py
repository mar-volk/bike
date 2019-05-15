import torch
import models
import train
import time
import pandas as pd
import numpy as np
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

dtype = torch.float
device = torch.device("cpu")

result_path = 'results/'
data_path = '../../data/'

# load data
parse_dates = ['dteday']
hour_x_train = pd.read_csv('../../data/train/hour_x.csv', parse_dates=parse_dates)
hour_x_val = pd.read_csv('../../data/val/hour_x.csv', parse_dates=parse_dates)
hour_y_train = pd.read_csv('../../data/train/hour_y.csv')
hour_y_val = pd.read_csv('../../data/val/hour_y.csv')

# preprocessing of predictors
hour_x_train['dteday'] = (hour_x_train['dteday'] - pd.to_datetime('2011-01-01')) / pd.Timedelta('1 days')
hour_x_val['dteday'] = (hour_x_val['dteday'] - pd.to_datetime('2011-01-01')) / pd.Timedelta('1 days')
numeric_features = ['dteday', 'temp', 'atemp', 'hum', 'windspeed']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)])

preprocessor.fit(hour_x_train)
x_train = preprocessor.transform(hour_x_train).todense()
x_val = preprocessor.transform(hour_x_val).todense()
pickle.dump(preprocessor, open('encoder.p', "wb"))  # Save encoder
print('Predictors prepared')

# Prepare targets
y_train = hour_y_train.values.astype(float)
y_val = hour_y_val.values.astype(float)
print('all data prepared')

# Test different weight decays
weight_decay_list = 10**np.linspace(-5, 3, 10)


val_loss_list = []
result_path_list = []
criterion = torch.nn.MSELoss()

for weight_decay in weight_decay_list:
    print('weight_decay: ' + str(weight_decay))

    # initialize model
    model = models.NNLinRegr2Layers('NNLinRegr2Layers', x_train.shape[1], d_h=100, d_out=1, p_drop=0, device='cpu')

    result_path = 'results_' + str(weight_decay) + '_/'

    # run training
    print('run model')
    start = time.time()
    training = train.TrainRegr(model, result_path, x_train, y_train, x_val, y_val)
    trained_model = training.run_train(1000, lr=0.01, batch_size=64, weight_decay=weight_decay)

    print('time [s]:')
    print(time.time()-start)

    # Get Loss
    x_val_torch = torch.tensor(x_val, device=model.device, dtype=model.dtype)
    y_val_torch = torch.tensor(y_val, device=model.device, dtype=model.dtype)#.squeeze()


    trained_model.eval()
    y_val_pred = trained_model(x_val_torch)#.squeeze()

    print(y_val_torch.shape)
    print(y_val_pred.shape)

    loss_test = ((y_val_pred - y_val_torch)**2).mean()
    print(loss_test)


    val_loss = criterion(y_val_pred, y_val_torch).item()
    print('val loss: ' + str(val_loss))

    val_loss_rev = criterion(y_val_torch, y_val_pred).item()
    print('val loss rev: ' + str(val_loss_rev))

    val_loss_list.append(val_loss)

    result_path_list.append(result_path)

res = pd.DataFrame()
res['weight_decay'] = weight_decay_list
res['validation loss'] = val_loss_list
res.to_csv('results_val_loss__.csv', index=False)
print(res)
