import torch
import models
import train
import time
import pandas as pd
import pickle
from prep import DataEncoder

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
enc = DataEncoder()
enc.fit(hour_x_train)
x_train = enc.transform(hour_x_train)
x_val = enc.transform(hour_x_val)
pickle.dump(enc, open('encoder.p', "wb"))  # Save encoder
print('Predictors prepared')

# Prepare targets
y_train = hour_y_train.values.astype(float)
y_val = hour_y_val.values.astype(float)
print('all data prepared')

# Test different batch-sizes
batch_size_list = [32, 64, 128, 256, 512, 1024]
weight_decay = 0

val_loss_list = []
result_path_list = []
criterion = torch.nn.MSELoss()

for batch_size in batch_size_list:
    print('batch_size: ' + str(batch_size))

    # initialize model
    model = models.LinRegr('LinRegr', x_train.shape[1], d_out=1, dtype=dtype, device=device)

    result_path = 'results_' + str(batch_size) + '_/'

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

    loss_test = ((y_val_pred - y_val_torch)**2).mean()
    print(loss_test)


    val_loss = criterion(y_val_pred, y_val_torch).item()
    print('val loss: ' + str(val_loss))

    val_loss_rev = criterion(y_val_torch, y_val_pred).item()
    print('val loss rev: ' + str(val_loss_rev))

    val_loss_list.append(val_loss)

    result_path_list.append(result_path)

res = pd.DataFrame()
res['batch_size'] = batch_size_list
res['validation loss'] = val_loss_list
res.to_csv('results_val_loss__.csv', index=False)
print(res)
