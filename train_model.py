'''
---------------------------------------------------
Devin Suy : https://devinsuy.com/
Dataset: https://www.kaggle.com/c/dogs-vs-cats/data
---------------------------------------------------

Train 3 layer sequential model for binary classification {cat, dog} of RGB images
'''

import sys
from numpy import load
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD
import os
 
# Select GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Home directory
DATA_DIR = "data/"

# Define and compile model
# Input is in the form of 200x200 pixel RGB images
cat_dog_classifer = Sequential([
    # Layer 1
    Conv2D(32, (3,3), input_shape=(200, 200, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    MaxPooling2D((2, 2)),

    # Layer 2
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    MaxPooling2D((2, 2)),

    # Layer 3
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_uniform'),

    # Binomial classification {cat, dog} single node output layer
    Dense(1, activation='sigmoid')
])
cat_dog_classifer.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=0.001, momentum=0.9))

# Process images by scaling the value of pixels between 0:1
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Full image dataset is RAM intensive to load directly, use iterators from API instead
train_data = datagen.flow_from_directory(
    DATA_DIR + 'train/', 
    class_mode='binary', 
    target_size=(200, 200),
    batch_size=64
)
train_size = len(train_data)
test_data = datagen.flow_from_directory(
    DATA_DIR + 'test/', 
    class_mode='binary', 
    target_size=(200, 200),
    batch_size=64
)
test_size = len(test_data)

# Begin training model
eval = cat_dog_classifer.fit_generator(
    generator=train_data, 
    steps_per_epoch=train_size,
    validation_data=test_data, 
    validation_steps=test_size, 
    epochs=20, 
    verbose=2
)

# Save model
cat_dog_classifer.save('cat_dog_classifer.h5')

# Output history of loss and accuracy during training
print("\nLoss\n----\n")
print(eval.history['loss'])
print("\nAccuracy\n--------")
print(eval.history['accuracy'])