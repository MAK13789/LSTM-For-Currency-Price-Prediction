import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
data_test = open("PKR Data Test.txt", "r")
data_train = open("PKR Data Train.txt", "r")
num_test = 5
num_train = 171
date_train = []
price_train = []
date_test = []
price_test = []
for i in range(num_train):
    temp_1 = data_train.readline()
    temp_2 = temp_1.split()
    temp_3 = temp_2[0]
    date_train.append(temp_3)
    temp_4 = temp_2[1]
    price_train.append(temp_4)  
for j in range(len(price_train)):
    price_train[j] = float(price_train[j])
combined_train = list(zip(date_train, price_train))
train = pd.DataFrame(combined_train, columns = ['Date', 'Price'])
train = train.iloc[::-1]
processed_price_train = train.iloc[:, 1:2].values
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_price_train = scaler.fit_transform(processed_price_train)
features_set = []  
labels = []  
for k in range(5, len(scaled_price_train)):
    features_set.append(scaled_price_train[k-5:k, 0])
    labels.append(scaled_price_train[k, 0])
features_set = np.array(features_set)
labels = np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
model = keras.models.Sequential([
    keras.layers.LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1],1)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=50, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=50, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=50),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=1)
])
model.summary()
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(features_set, labels, epochs = 75, batch_size = 4) 
for a in range(num_test):
    temp_1 = data_test.readline()
    temp_2 = temp_1.split()
    temp_3 = temp_2[0]
    date_test.append(temp_3)
    temp_4 = temp_2[1]
    price_test.append(temp_4)
for b in range(len(price_test)):
    price_test[b] = float(price_test[b])
combined_test = list(zip(date_test, price_test))
test = pd.DataFrame(combined_test, columns = ['Date', 'Price'])
test = test.iloc[::-1]
processed_price_test = test.iloc[:, 1:2].values
data_total = pd.concat((train['Price'], test['Price']), axis = 0)
test_inputs = data_total[num_train - num_test - 5:].values
test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)
test_features = []
for c in range(5, 10):
    test_features.append(test_inputs[c-5:c, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
plt.figure(figsize=(10,6))  
plt.plot(processed_price_test, color='blue', label='Actual Rupee Price')  
plt.plot(predictions , color='red', label='Predicted Rupee Price')  
plt.title('Rupee Price Prediction')  
plt.xlabel('Date')  
plt.ylabel('Rupee Price')  
plt.legend()  
plt.show()
x = data_total[num_train + num_test - 5:].values
x = x.reshape(-1, 1)
x = scaler.transform(x)
y = []
y.append(x[0:5,0])
y = np.array(y)
y = np.reshape(y, (y.shape[0], y.shape[1], 1))
prediction = model.predict(y)
prediction = scaler.inverse_transform(prediction)
print ("Price for Tomorrow:")
print (prediction) 
