import random
import pandas
from pandas import set_option
from pandas import DataFrame
from pandas import concat
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras.optimizers import adam, SGD
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
from sklearn.metrics import mean_squared_error


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
         agg.dropna(inplace=True)
    return agg

###########################################################################################################

dataframe_all = pandas.read_csv('./international-airline-passengers.csv', 'r', delimiter=";" , usecols=[1], header=0)
plt.plot(dataframe_all)
plt.legend(['original'], loc='upper left')
plt.title('Airline passengers')
plt.show()

l_orig = len(dataframe_all)

look_back = 3
look_in_front = 1

# prepare time series to supervised learning task
dataframe = series_to_supervised(dataframe_all, look_back, look_in_front, dropnan=True)

show_data=False
if (show_data):
    set_option('display.max_columns', 50)
    print()
    print('reframed dataset:')
    print(dataframe.head())
    print()
    input("Press Enter to continue...")

data_airlineX = dataframe.iloc[:, 0:look_back].values
data_airlineY = dataframe.iloc[:, -look_in_front:].values


# normalize the dataset
scaler = MinMaxScaler()
data_airlineX = scaler.fit_transform(data_airlineX)
data_airlineY = scaler.fit_transform(data_airlineY)

# With time series data, the sequence of values is important.
# A simple method that we can use is to split the ordered dataset into train and test datasets.

# The code below calculates the index of the split point and separates the data into the training datasets
# with 88% of the observations that we can use to train our model,
# leaving the remaining 12% for testing the model.

# split into train and test sets
train_size = int(len(data_airlineX) * 0.88)
test_size = len(data_airlineX) - train_size

trainX, testX = data_airlineX[0:train_size,0:look_back], data_airlineX[train_size:len(data_airlineX),0:look_back]
trainY, testY = data_airlineY[0:train_size,-look_in_front:], data_airlineY[train_size:len(data_airlineY),-look_in_front:]

print(trainX.shape)
print(trainY.shape)


#The network expects the input data (X) to be provided with a specific array structure
# in the form of: [samples, time steps, features].

#Currently, our data is in the form: [samples, features] and we are framing the problem
# as one time step for each sample. We can transform the prepared train and test input data
# into the expected structure using numpy.reshape() as follows:

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

np.random.seed(7)

# create and fit the  network
model = Sequential()
model.add(SimpleRNN(10, input_shape=(trainX.shape[1], 1),activation='linear'))
model.add(Dense(look_in_front))

opt=adam(lr=0.0005)
model.compile(loss='mean_squared_error', optimizer=opt)

history=model.fit(trainX, trainY, epochs=250, batch_size=5, verbose=1, validation_data=(testX,testY))

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# calculate root mean squared error for Train and Prediction
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting

trainPredictPlot = np.empty([l_orig])
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:look_back+len(trainPredict)] = trainPredict.T[look_in_front-1]

# shift test predictions for plotting

testPredictPlot = np.empty([l_orig])
testPredictPlot[:] = np.nan
length=len(testPredict.T[0])
testPredictPlot[l_orig-length:] = testPredict.T[look_in_front-1]

# plot results

plt.plot(dataframe_all)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(['original', 'train', 'prediction'], loc='upper left')
plt.title('Airline passengers')
plt.show(block=False)


# plot history for loss

plt.figure(30)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

