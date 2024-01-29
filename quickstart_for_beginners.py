################################################################################
## Machine Learning Model that Classifies Images
## Based on tutorial:
## https://www.tensorflow.org/tutorials/quickstart/beginner#load_a_dataset
################################################################################

import tensorflow as tf

# Check TensorFlow version
print(tf.__version__)

# Simple computation
hello = tf.constant('Hello, TensorFlow!')
tf.print(hello)

# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a tf.keras.Sequential model (machine learning model)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Print logits or log-odds scores for each class
predictions = model(x_train[:1]).numpy()
print(predictions)

# Convert logits to probabilities for each class
tf.nn.softmax(predictions).numpy()

# Define a loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Cofigure and compile model using Keras Model.compile
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Run and train model
model.fit(x_train, y_train, epochs=5)

# Check model's performance
model.evaluate(x_test,  y_test, verbose=2)