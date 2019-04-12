
# coding: utf-8

# ## Logic Based FizzBuzz Function [Software 1.0]

# In[45]:


import numpy as np
#!pip install tensorflow
import tensorflow as tf


# In[46]:


import pandas as pd

def fizzbuzz(n):
    
    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format

# In[47]:


def createInputCSV(start,end,filename):
    
    # Why list in Python?
    # A list in python can be used to store items of different types. We are using Lists because they are mutable i.e., they can
    # be changed once declared. Lists are very flexible. Lists are useful for storing lot of values and also to iterate over those
    # values or to modify those values.
    
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    # Why are we using machines in the first place? - because they are able to store and process knowledge at faster rates. 
    # Similar to humans who learn from context and are able to predict the value of the unknown, the machine learning models
    # can do the same when they see enough relevant data. So the role of training data is to to train our machine learning models.
    # after training the model on a particular training data, if satisfactory we can use the model to predict the outputs for different inputs.
    
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe?
    # Dataframe is used to build a kind of table/database to store values of different fields i.e., the data is stored as rows and 
    # columns. Each row contains measurements or values of an instance, while each column contains data of a specific variable.
    
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[48]:


def processData(dataset):
    
    # Why do we have to process?
    # We have to process to encode the data. We need to turn each input into a vector of 'activations' (joelgrus.com). One way is to convert the input to binary.
    # The goal of processing in our case is to convert input to binary form and the output(label) to discrete values. So our input will be binary and output will be numbers 0, 1, 2, 3
    # The neural network inputs represent a kind of "intensity" so if we have larger value inputs they have greater intensity. So, to avoid this difference in magnitudes we do 1-of-k encoding which would ensure that training set isn't biased towards small or large numbers.
    # It also increases the number of features as a single integer is being represented by 10 bits.
    
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[49]:


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        # Because in the training data we have 900 values and to convert all these values into binary form we need 10 bits because,
        # 2**10 = 1024. 10 bits can be used to represent 1024 values where as 9 bits(2**9) can only represent 512 values.
        
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


# In[50]:


from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)


# ## Model Definition

# In[51]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
import keras

import numpy as np

input_size = 10
drop_out = 0.2
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 4

def get_model():
    
    # Why do we need a model?
    # The core abstraction of every model in keras is the notion of layers. Models are used to define our neural networks.
    
    # Why use Dense layer and then activation?
    # In general both the dense layer and the activation in the same line but the advantage of using dense layer first and then
    # activation is the outputs of the dense layer(last layer) could be retrieved before activation. 
    
    # Why use sequential model with layers?
    # Because we have single-input and single-output. Sequential model is also simple and used for implementing sequence of layers using model.add
    # Even though the Functional API can also be used to do this, sequential model is simple and is specifically designed for these simple cases
    
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # Why dropout?
    # dropout is a (regularization) technique of ignoring randomly selected neurons during training.
    # The effect is, the network becomes less sensitive to specific weight of the neurons and this results in a network that is more generalizable.
    # The main goal is to prevent overfitting the training data.
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?
    # 1. Because it is a multiple classification problem and sigmoid and ReLU can't help much in this case.
    # 2. Softmax function outputs a probability distribution, it tells u the probability that any of the classes are true which is more desirable. 
    # 3. Its derivatie is computationally cheap to calcuate and useful during backpropogation.
    # 4. The exp in the softmax function roughly cancels out the log in the cross-entropy loss causing the loss to be roughly linear in z_i.
    #    This leads to a roughly constant gradient, when the model is wrong, allowing it to correct itself quickly. Thus, a wrong saturated softmax does not cause a vanishing gradient. (from: https://stackoverflow.com/questions/17187507/why-use-softmax-as-opposed-to-standard-normalization)
    
    model.summary()
    
    # Why use categorical_crossentropy?
    # Since, the output(the dependent variable) has more than 2 categories in output we use categorical_crossentropy. The categoricol cross-entropy aims to maximize the softmax output of the correct label.
    # Because, the output from softmax is a probability value between 0 and 1. The cross entropy can be used with other activation functions as well but it is better suited(and gives better results) with softmax
    # Also, the derivative of cross entropy loss with Softmax is simple
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# # <font color='blue'>Creating Training and Testing Datafiles</font>

# In[52]:


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# # <font color='blue'>Creating Model</font>

# In[53]:


model = get_model()


# # <font color = blue>Run Model</font>

# In[54]:


validation_data_split = 0.2 #takes last 20% of the data from the training set as validation data

num_epochs = 1500
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=0, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# # <font color = blue>Training and Validation Graphs</font>

# In[55]:


get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # <font color = blue>Testing Accuracy [Software 2.0]</font>

# In[56]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# In[57]:


wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "rsunkara")
testDataLabel.insert(0, "50292191")

testDataInput.insert(1, "rsunkara")
testDataLabel.insert(1, "50292191")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

