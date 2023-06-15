#!/usr/bin/env python
# coding: utf-8

# Build a DNN with five hidden layers of 100 neurons each, He initialization, and the ELU activation function

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()

model.add(layers.InputLayer(input_shape=(0,)))

for _ in range(5):
    model.add(layers.Dense(100, activation='elu', kernel_initializer='he_normal'))

model.add(layers.Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Using Adam optimization and early stopping, try training it on MNIST but only on digits 0 to 4, as we will use transfer learning for digits 5 to 9 in the next exercise. You will need a softmax output layer with five neurons, and as always make sure to save checkpoints at regular intervals and save the final model so you can reuse it later.

# In[ ]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, callbacks

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_filter = (y_train <= 4)
test_filter = (y_test <= 4)
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    *[layers.Dense(100, activation='elu', kernel_initializer='he_normal') for _ in range(5)],
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = callbacks.ModelCheckpoint('mnist_dnn_checkpoint.h5', save_best_only=True)
early_stopping_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)


model.fit(x_train, tf.keras.utils.to_categorical(y_train, 5), 
          validation_data=(x_test, tf.keras.utils.to_categorical(y_test, 5)), 
          epochs=100, callbacks=[checkpoint_cb, early_stopping_cb])

model.save('mnist_dnn_final.h5')


# Tune the hyperparameters using cross-validation and see what precision you can achieve.

# Split the data: Divide your dataset into two parts: a training set and a validation set. The training set will be used to train the model, and the validation set will be used for hyperparameter tuning.
# 
# Define the hyperparameters: Determine which hyperparameters you want to tune. Examples include learning rate, regularization strength, the number of hidden layers, or the number of decision trees in a random forest.
# 
# Choose a hyperparameter search space: Define a range or set of values for each hyperparameter. This search space will be explored during the tuning process.
# 
# Choose a cross-validation strategy: Decide on the number of folds for cross-validation. Common choices include 5-fold or 10-fold cross-validation.
# 
# Perform hyperparameter tuning: Use a search algorithm, such as grid search or randomized search, to explore the hyperparameter search space. For each combination of hyperparameters, train the model on the training set and evaluate its performance using the validation set and a chosen evaluation metric, such as precision.
# 
# Select the best hyperparameters: Identify the combination of hyperparameters that resulted in the highest precision on the validation set.
# 
# Evaluate on a test set: After selecting the best hyperparameters, train a new model using these hyperparameters on the entire training set. Finally, evaluate the model's performance on a separate test set to get an unbiased estimate of its precision.

# Now try adding Batch Normalization and compare the learning curves: is it converging faster than before? Does it produce a better model?
# 

# Batch Normalization is a technique commonly used in deep neural networks to improve training efficiency and generalization. It normalizes the inputs to each layer by subtracting the batch mean and dividing by the batch standard deviation.
# 
# Set up the experiment: Start by preparing your dataset and defining the model architecture, including the addition of Batch Normalization layers.
# 
# Train the model without Batch Normalization: Train the model on your dataset without using Batch Normalization. Record the training loss and evaluation metrics (e.g., precision) at regular intervals during training. Plot the learning curves based on these recorded values.
# 
# Train the model with Batch Normalization: Modify the model to include Batch Normalization layers after appropriate layers (e.g., fully connected or convolutional layers). Train the model again on the same dataset, recording the training loss and evaluation metrics as before. Plot the learning curves for this version of the model.
# 
# Compare the learning curves: Analyze and compare the learning curves of the two models. Look for differences in convergence speed, stability, and generalization performance. Specifically, observe if the model with Batch Normalization achieves lower training loss, faster convergence, and higher precision compared to the model without Batch Normalization.
# 
# Evaluate the final model: After training, evaluate the performance of the model with Batch Normalization on a separate test set. Calculate relevant evaluation metrics (e.g., precision, accuracy, recall, F1 score) and compare them with the performance of the model without Batch Normalization.
#     
#     

# Is the model overfitting the training set? Try adding dropout to every layer and try again. Does it help?
# 

# Overfitting occurs when a machine learning model becomes too specialized to the training data, resulting in poor generalization to new, unseen data. Dropout is a regularization technique commonly used to combat overfitting in neural networks. It randomly sets a fraction of input units to 0 during training, which helps prevent co-adaptation of neurons and encourages the model to learn more robust and generalizable representations.
# 
# Set up the experiment: Prepare your dataset and define the model architecture, including adding dropout layers after each layer in the network.
# 
# Train the model without dropout: Train the model on your dataset without using dropout. Record the training loss and evaluation metrics (e.g., precision) at regular intervals during training. Plot the learning curves based on these recorded values.
# 
# Train the model with dropout: Modify the model to include dropout layers after each layer (or selectively choose specific layers). Train the model again on the same dataset, recording the training loss and evaluation metrics as before. Plot the learning curves for this version of the model.
# 
# Compare the learning curves: Analyze and compare the learning curves of the two models. Look for differences in convergence speed, stability, and generalization performance. Specifically, observe if the model with dropout reduces overfitting, improves convergence, and enhances the model's precision compared to the model without dropout.
# 
# Evaluate the final model: After training, evaluate the performance of the model with dropout on a separate test set. Calculate relevant evaluation metrics (e.g., precision, accuracy, recall, F1 score) and compare them with the performance of the model without dropout.
# 

# Create a new DNN that reuses all the pretrained hidden layers of the previous model, freezes them, and replaces the softmax output layer with a new one.
# 

# Load the pretrained model: Load the weights and architecture of the previous model that contains the pretrained hidden layers. This can typically be done using the same machine learning library or framework you used for training the initial model.
# 
# Freeze the pretrained hidden layers: Iterate through the layers of the loaded model and set their trainable attribute to False. This ensures that the weights of the pretrained layers are not updated during the training of the new model. By freezing these layers, you retain the learned representations from the previous model while focusing on training the new output layer.
# 
# Create a new output layer: Replace the original softmax output layer with a new output layer that suits your specific task. The number of neurons in the output layer should match the number of classes or targets in your problem. Depending on the task, you may need to adjust the activation function or include additional layers after the new output layer.
# 
# Compile the new model: Compile the new model by specifying the optimizer, loss function, and any additional metrics you want to track during training and evaluation. Use the same or a similar configuration to the one used for training the original model.
# 
# Train the new model: Train the new model using your dataset. Since the pretrained hidden layers are frozen, only the weights of the new output layer will be updated during training. You can use techniques such as early stopping, learning rate scheduling, or regularization to optimize the training process.
# 
# Evaluate the new model: After training, evaluate the performance of the new model on a separate test set. Calculate relevant evaluation metrics (e.g., precision, accuracy, recall, F1 score) to assess its performance. Compare the results with the previous model to determine if reusing the pretrained hidden layers has benefited the new model.

# Train this new DNN on digits 5 to 9, using only 100 images per digit, and time how long it takes. Despite this small number of examples, can you achieve high precision?
# 

# Data Preparation: Collect or obtain a dataset consisting of images of digits 5 to 9. Ensure that each digit has only 100 images available. Split the dataset into training and test sets, maintaining the class distribution.
# 
# Model Architecture: Define a deep neural network architecture for the task. Reuse the pretrained hidden layers from the previous model and freeze their weights. Replace the softmax output layer with a new layer that corresponds to the five digits (5 to 9) in the classification task.
# 
# Freeze Pretrained Layers: Set the trainable parameter of the pretrained hidden layers to False so that their weights are not updated during training.
# 
# Compile the Model: Compile the model by specifying the loss function (e.g., categorical cross-entropy) and an optimizer (e.g., Adam) for training.
# 
# Train the Model: Train the model using the training set. Since we have a small number of examples, consider using techniques such as data augmentation to increase the effective dataset size. Monitor the training process, including the training time, loss, and any chosen evaluation metrics such as precision, during the training iterations.
# 
# Evaluate the Model: Evaluate the trained model on the test set to assess its performance. Calculate the precision and other relevant evaluation metrics to determine the model's accuracy and generalization ability.
# 
# 

# Try caching the frozen layers, and train the model again: how much faster is it now?
# 

# Caching the frozen layers can significantly speed up the training process, as it eliminates the need to compute gradients and update the weights of these layers during each training iteration.
# 
# Load the Pretrained Model: Load the pretrained model with the frozen hidden layers.
# 
# Freeze the Layers: Set the trainable parameter of the pretrained hidden layers to False to ensure they are not updated during training.
# 
# Cache the Hidden Layers: Forward pass a subset of the training data through the frozen layers and store the output activations (i.e., the cached values) of these layers. These cached activations will be reused during the subsequent training iterations.
# 
# Replace the Output Layer: Replace the softmax output layer with a new layer that corresponds to the classification task for digits 5 to 9.
# 
# Compile the Model: Compile the model by specifying the loss function and optimizer for training the new output layer.
# 
# Train the Model: Train the model using the training set. During each training iteration, use the cached values of the frozen layers as inputs and only update the weights of the new output layer. Monitor the training process, including training time, loss, and any chosen evaluation metrics.

# Try again reusing just four hidden layers instead of five. Can you achieve a higher precision?
#     

# Reusing four hidden layers instead of five may or may not result in a higher precision. The impact on precision depends on various factors such as the complexity of the task, the importance of the information captured by the fifth hidden layer, and the quality of the pretrained model.
# 
# 
# Data Preparation: Prepare the dataset containing images of digits 5 to 9, with 100 images per digit. Split the dataset into training and test sets, preserving the class distribution.
# 
# Model Architecture: Define a deep neural network architecture for the task, reusing only the four pretrained hidden layers from the previous model. Replace the softmax output layer with a new layer corresponding to the five digits (5 to 9) in the classification task.
# 
# Freeze Four Hidden Layers: Set the trainable parameter of the four pretrained hidden layers to False, ensuring that their weights remain unchanged during training.
# 
# Compile the Model: Compile the model by specifying the loss function (e.g., categorical cross-entropy) and an optimizer (e.g., Adam) for training.
# 
# Train the Model: Train the model using the training set. Monitor the training process, including the training time, loss, and any chosen evaluation metrics (e.g., precision).
# 
# Evaluate the Model: Evaluate the trained model on the test set to assess its precision and overall performance.

# Now unfreeze the top two hidden layers and continue training: can you get the model to perform even better?
# 

# Unfreezing the top two hidden layers and continuing training allows the model to adapt and fine-tune these layers specifically for the new task. This additional training can potentially lead to improved performance compared to using only the pretrained layers.
# 
# Load the Pretrained Model: Load the pretrained model with the four hidden layers frozen.
# 
# Unfreeze the Top Two Hidden Layers: Set the trainable parameter of the top two hidden layers to True, allowing their weights to be updated during training. Keep the lower two hidden layers frozen.
# 
# Compile the Model: Compile the model by specifying the loss function and optimizer for training.
# 
# Continue Training: Resume training the model using the training set. The weights of the unfrozen layers will now be updated during the training process, while the frozen layers will retain their previously learned representations.
# 
# Monitor Training Process: Monitor the training process, including the training time, loss, and evaluation metrics such as precision, to assess the model's performance during training.
# 
# Evaluate the Model: Evaluate the model on the test set to assess its performance, precision, and generalization ability.

# In this exercise you will build a DNN that compares two MNIST digit images and predicts whether they represent the same digit or not. Then you will reuse the lower layers of this network to train an MNIST classifier using very little training data. Start by building two DNNs (let’s call them DNN A and B), both similar to the one you built earlier but without the output layer: each DNN should have five hidden layers of 100 neurons each, He initialization, and ELU activation. Next, add one more hidden layer with 10 units on top of both DNNs. To do this, you should use TensorFlow’s concat() function with axis=1 to concatenate the outputs of both DNNs for each instance, then feed the result to the hidden layer. Finally, add an output layer with a single neuron using the logistic activation function.
# 

# Import Libraries: Import the necessary libraries, including TensorFlow.
# 
# Define DNN Architectures: Define the architectures for DNN A and DNN B. Each DNN should have five hidden layers of 100 neurons each, He initialization, and ELU activation.

# In[33]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.activations import elu

input_shape = (900,) 

input_a = Input(shape=input_shape)
hidden_a = Dense(100, activation=elu, kernel_initializer='he_normal')(input_a)
for _ in range(4):
    hidden_a = Dense(100, activation=elu, kernel_initializer='he_normal')(hidden_a)
dnn_a = tf.keras.Model(inputs=input_a, outputs=hidden_a)

input_b = Input(shape=input_shape)
hidden_b = Dense(100, activation=elu, kernel_initializer='he_normal')(input_b)
for _ in range(4):
    hidden_b = Dense(100, activation=elu, kernel_initializer='he_normal')(hidden_b)
dnn_b = tf.keras.Model(inputs=input_b, outputs=hidden_b)

concatenated_output = Concatenate(axis=1)([dnn_a.output, dnn_b.output])

hidden_layer = Dense(10, activation=elu)(concatenated_output)

output_layer = Dense(1, activation='sigmoid')(hidden_layer)

model = tf.keras.Model(inputs=[dnn_a.input, dnn_b.input], outputs=output_layer)


# Split the MNIST training set in two sets: split #1 should containing 55,000 images, and split #2 should contain contain 5,000 images. Create a function that generates a training batch where each instance is a pair of MNIST images picked from split #1. Half of the training instances should be pairs of images that belong to the same class, while the other half should be images from different classes. For each pair, the training label should be 0 if the images are from the same class, or 1 if they are from different classes.
# 

# In[5]:


import numpy as np
import random

def generate_training(split1, num_pairs):
    images = split1.images
    labels = split1.labels
    
    pairs = []
    target_labels = []
    
    for _ in range(num_pairs // 2):
        label = random.choice(range(10))
        indices = [i for i, l in enumerate(labels) if l == label]
        pair = random.sample(indices, 2)
        pairs.append(pair)
        target_labels.append(0)
        
        label1, label2 = random.sample(range(10), 2)
        indices1 = [i for i, l in enumerate(labels) if l == label1]
        indices2 = [i for i, l in enumerate(labels) if l == label2]
        pair = [random.choice(indices1), random.choice(indices2)]
        pairs.append(pair)
        target_labels.append(1)
    
    return pairs, target_labels


# In[9]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


# In[10]:


(X_train, y_train), (_, _) = mnist.load_data()

split1_images, _, split1_labels, _ = train_test_split(X_train, y_train, train_size=55000, random_state=42)

print("Split1 images shape:", split1_images.shape)
print("Split1 labels shape:", split1_labels.shape)


# In[ ]:





# In[ ]:





# In[ ]:




