#Generation of standard initial weights and biases
from numpy.random import seed
seed(101)
import tensorflow as tf
tf.random.set_seed(101)

#import appropriate libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Dropout, GaussianNoise
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import the data
df_inputs = pd.read_excel('CollectiveBreastFile.xlsx', sheet_name = 'All_in')
df_outputs = pd.read_excel('CollectiveBreastFile.xlsx', sheet_name = 'All_out')

#arranging the dimensions for NN
df_inputs = df_inputs.T
df_outputs = df_outputs.T

#splitting the data to training and testing
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(df_inputs, df_outputs, test_size=0.1, random_state=101, shuffle=True)

#Parameters used in the network
PIs = 0
number_of_genes = 142 #it's actually the number of glycans
number_of_glycans = 1 #this is the indicator whether the cell is a normal cell or not
iterations = 3000

#Build the network
#The #186 system
model = Sequential([
    #GaussianNoise(0.1,
    #    input_shape= (number_of_genes + PIs,)),
    Dense(units = 24,
          activation= 'relu',
          input_shape= (number_of_genes + PIs,)),
    #Dropout(0.2),
    #Dense(units = 10,
    #      activation='sigmoid'),
    Dense(units = number_of_glycans,
          activation = 'sigmoid')])

model.compile(
    optimizer = Adam(lr=0.01),
    loss = 'mean_squared_error',
    metrics = ['accuracy'])

model.fit(
    train_inputs,
    train_outputs,
    epochs = iterations,
    batch_size = 32,
    shuffle = False,)

print(model.evaluate(
    test_inputs,
    test_outputs,
    batch_size = 32,
    verbose = 1))


#Calculate the labels for the test set
labels_test = model.predict(test_inputs)
labels_test = pd.DataFrame(data = labels_test)

#Calculate again the labels for the train set
labels_train = model.predict(train_inputs)
labels_train = pd.DataFrame(data = labels_train)

#bringing it back to the normal structure
labels_testT = labels_test.T
test_outputsT = test_outputs.T

labels_trainT = labels_train.T
train_outputsT = train_outputs.T

#giving labelsT the correct column names
labels_testT.columns = test_outputsT.columns
labels_trainT.columns = train_outputsT.columns

print(labels_testT.T)
print(labels_trainT.T)

"""

#Join the data for exporting model fitting and predictions
comparison_test = test_outputsT.join(labels_testT, how='left', lsuffix='_left', rsuffix='_right')
comparison_train = train_outputsT.join(labels_trainT, how='left', lsuffix='_left', rsuffix='_right')

writer = pd.ExcelWriter('Comparison.xlsx', engine='xlsxwriter')
comparison_test.to_excel(writer, sheet_name='Test fitting')
comparison_train.to_excel(writer, sheet_name='Train fitting')
writer.save()


#Preparing a graph showing model fitting
#Create numpy arrays to include all the model predictions and experimental data
model_test = np.zeros((number_of_glycans, len(test_inputs)))
data_test = np.zeros((number_of_glycans, len(test_inputs)))

#assign the values
for i in range (len(test_inputs)):
    model_test[:,i] = test_outputsT[test_outputsT.columns[i]]
    data_test[:,i] = labels_testT[labels_testT.columns[i]]

#convert to the right dimensions
model_test = model_test.T
data_test = data_test.T

#preparing the y=x
linearx = np.linspace(0,1,10)
lineary = linearx

#plotting
plt.scatter(data_test, model_test)
plt.xlabel('experimental data')
plt.ylabel('model data')
plt.plot(linearx, lineary, 'r')
plt.show()


"""