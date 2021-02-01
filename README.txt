---------------------------------
Devin Suy : https://devinsuy.com/
---------------------------------

--------
Summary:
--------
- Setup Nvidia CUDNN for GPU utilization with keras using tensorflow as backend library

- Using cat and dog image dataset: https://www.kaggle.com/c/dogs-vs-cats/data
- Trained simple model consisting of 3 layers of 2D convolutional layers for binary classification {cat, dog}
- After training tested model on 20 new images
	- Correctly classified cat and dog images with accuracy ~ 0.8, ALL misclassifications were images of cats

- Insightful project for familiarizing with development in keras environment and exposure to deep learning theory 
- Deep learning topics to continue exploring: 
	- Dropout regularization
	- Data augmentation
	- Transfer learning
	- Back propagation


---------------
Included Files:
---------------
- "data/"
- "predict_result_imgs/"

- "train_model.py"
- "test_model.py"
- "cat_dog_classifer.h5"
- "training_logs.txt"
- "deep_learning_notes.txt"


-----------------------------------------
Training Results (See training_logs.txt):
-----------------------------------------
- 18698 training images, 6302 validation images
- Optimizer used: Stochastic Gradient Descent
- Loss used: Binary Crossentropy

- After 20 epochs
	- Accuracy: 0.9499, val_accuracy: 0.7882
	- loss: 0.1393, val_loss: 0.5348


-------------------
Prediction Results:
-------------------
- 20 images to predict on: 10 cat, 10 dog

- 16 classified correctly, 4 incorrect (images 3, 11, 14, 18)
	- All 4 incorrect classifications were of cat images incorrectly predicted as dog
	- Possibly notable: image 11 of cat is very dark and angled strangely, image 14 contains two cats

- Accuracy: 0.8


-------------------
Utilized Resources:
-------------------
Setup:
- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
- https://medium.com/@ab9.bhatia/set-up-gpu-accelerated-tensorflow-keras-on-windows-10-with-anaconda-e71bfa9506d1


Keras:
- https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit_generator
- https://keras.io/api/layers/activations/
- https://keras.io/api/layers/initializers/
- https://keras.io/api/layers/convolution_layers/convolution2d/


Theory:
- https://computersciencewiki.org/index.php/Max-pooling_/_Pooling#:~:text=Max%20pooling%20is%20a%20sample,in%20the%20sub%2Dregions%20binned
- https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480
- https://missinglink.ai/guides/keras/using-keras-flatten-operation-cnn-models-code-examples/
- https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
- https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
- https://ruder.io/optimizing-gradient-descent/