import tensorflow as tf
import os
import pandas as pd
import numpy as np

# to solve issue "AttributeError: module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface'", only shows up when using internal APIs
# from tensorflow.python.keras.engine import data_adapter 
# def _is_distributed_dataset(ds):
#     return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
# data_adapter._is_distributed_dataset = _is_distributed_dataset

zip_path = tf.keras.utils.get_file(
    # origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    origin='file://'+os.path.dirname(os.path.abspath(__file__))+'/data/jena_climate_2009_2016.csv.zip', # finding file in same level directory as current file. idk if this is the best way to do this, but it works. 
    fname='jena_climate_2009_2016.csv.zip',
    extract=True
)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
df = df[5::6] # start from fifth element, take every sixth element
print(df.head())
print(df.shape)
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df[:26]
temp = df['T (degC)']
temp.plot()
# matrix where each row is input with corresponding label
# for forecasting, every 5 hours, predict next hour
# input: [[[1], [2], [3], [4], [5]]] output: [6]
# next row: [[[2], [3], [4], [5], [6]]] [7] why can we use the predicted data for next prediction though?
# supervised learning
# every individual input is in a list because if we want to include mutliple variables, we would add to those lists. input is a 3d tensor, but functionally a matrix with 1 variable for input
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy() #convert dataframe to numpy
    X = [] # input
    y = [] # output
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]] #
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)
WINDOW_SIZE = 5
X, y = df_to_X_y(temp, WINDOW_SIZE)
X.shape, y.shape #dimensions

# next, split into TRAIN, TEST, and VALIDATION sets

X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]
# can print the shapes to verify if you want
# majority of data goes to training, then smaller portions go to validationa nd testing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint # for saving models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
# anyone using this will probably have syntax errors for tensorflow.keras imports. tried to fix this by using tensorflow.python.keras but this brings up too many other issues. program works with the syntax errors. 

model1 = Sequential()
model1.add(InputLayer((5,1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

current_path = os.path.dirname(os.path.abspath(__file__))
model_path = current_path+'/model1/model1.keras'
cp = ModelCheckpoint(model_path, save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()]) # learning rate is how fast the model tries to decrease the loss
# saves best model to model1/ directory

model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])
# fit the model. measure performance on validation set. epochs is number of runs through the training dataset, callback calls back cp to check if we want to save the model depending on if validation loss is higher or lower. 

# after compiling..
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# Enable interactive mode
plt.ion()

plots_directory = current_path+'/plots'
model1 = load_model(model_path) # load back the model with lowest validation loss
# make pandas dataframe for TRAIN, TEST, and VALIDATION.
train_predictions = model1.predict(X_train).flatten() # flatten gets rid of inner/redundant brackets/dimensions
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})

plt.figure()
plt.plot(train_results['Train Predictions'][:100], label='Train Predictions')
plt.plot(train_results['Actuals'][:100], label='Actuals')
plt.legend()
plt.savefig(plots_directory+'/train_predictions.png')

# see how well model performs on validation data
val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions': val_predictions, 'Actuals': y_val})

plt.figure()
plt.plot(val_results['Val Predictions'][:100], label='Val Predictions')
plt.plot(val_results['Actuals'][:100], label='Actuals')
plt.legend()
plt.savefig(plots_directory+'/val_predictions.png')

#test data
test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})

plt.figure()
plt.plot(test_results['Test Predictions'][:100], label='Test Predictions')
plt.plot(test_results['Actuals'][:100], label='Actuals')
plt.legend()
plt.savefig(plots_directory+'/test_predictions.png')
