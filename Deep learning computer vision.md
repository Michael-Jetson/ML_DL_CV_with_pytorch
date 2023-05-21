## 1.  Keras Model Life-Cycle

1. Define Network
2. Compile Network
3. Fit Network
4. Evaluate Network
5. Make Predictions

<img src="assets/image-20230517190956258.png" alt="image-20230517190956258" style="zoom:67%;" />

### Step 1. Define Network

~~~python
model = Sequential()
model.add(Dense(2))
~~~

planb

~~~python
layers = [DEnse(2)]
model = Sequential(layers)
~~~

The first layer in the network must define the number of inputs to expect.

For a Multilayer Perceptorn model this is specified by the input_dim attribute

For activation function

![image-20230517193236988](assets/image-20230517193236988.png)

### Step 2. Compile Network

Thing of compilation as a precompute step for your network.

~~~pytho
model.compile(optimizer='sgd', loss='mean_squard_error')
~~~

planb

~~~python
algorithm=SGD(lr=0.1, momentum=0.3)
model.compile(optimizer=algorithm, loss='mean_squared_error')
~~~

loss functions

![image-20230517194205940](assets/image-20230517194205940.png)

optimization algorithm(the most common----stochastic gradient decsent（随机梯度下降法）)

![image-20230517200834298](assets/image-20230517200834298.png)

metricst

~~~python
model.compile(optimizer='sgd', loss='mean_squard_error', metrics=['accuracy'])
~~~

### Setp 3. Fit Network

~~~python
history=model.fit(X,y,batch_size=0,epochs=100,verbose=0)
~~~

### Step 4. Evaluate Network

~~~python
loss, accuracy = model.evaluate(X, y)
~~~

~~~python
loss, accuracy = model.evaluate(X, y, verbose=0)
~~~

### Step 5. Make Predictions

~~~python
predictions = model.predict(X)
~~~

The predictions will be returned in the format provided by the output layer of the network.

regression problem, provided by a linear activation funvtion.

binary classification problem, an array of probabilities for the first class.

multiclass classification problem, in the form of an array of probabilities(assuming a one hot encoded output variable)

~~~python
predictions = model.predict_classes(X)
~~~

And

~~~python
predictions = model.predict(X, verbose=0)
~~~

