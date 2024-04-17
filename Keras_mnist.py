import numpy as np

np.random.seed(1) 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()
from matplotlib import pyplot as plt
from PIL import Image
import random

plt.close('all')
plt.style.use("ggplot")

# get list of randomly selected images and create their labels as top probabilities
def get_random_images(images, subset_indices=None, n_images=10, probs=None, n_probs=None):
  if subset_indices is None: # if no subset_indices provided, then use all
      subset_indices = range(0,len(images))
  selected_indices = random.choices(subset_indices, k=n_images)

  imgs = []
  labels = []
  for imgidx in selected_indices:
    img = images[imgidx]
    if img.max()<=1.0: 
      img = img*255
    if np.ndim(img)==3 and img.shape[2]==1:
      img = np.reshape(img,(img.shape[0],img.shape[1]))
    img = Image.fromarray(img)
    imgs.append(img)

    if (probs is not None) and (n_probs is not None):
      top_probs = np.argsort(probs, axis=1)[:,:-(n_probs+1):-1]
      label_prob = ""
      for i in range(n_probs):
        label_prob += f"{top_probs[imgidx,i]} ({probs[imgidx,top_probs[imgidx,i]]:.3f}) "
      
      labels.append(label_prob)

  return imgs, labels

# plots images with labels
def show_images(ims, figsize=(12,6), rows=1, interp=False, titles=None, suptitle=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i])
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    if suptitle:
      f.suptitle(suptitle,fontweight="bold")

    f.tight_layout(rect=[0, 0, 1, 0.95])

# display randomly selected images
n_images = 20

imgs, labels = get_random_images(X_test/255, n_images=n_images)
show_images(imgs, figsize=(5,4), rows=4, suptitle=str(n_images) + " randomly selected images")

# Reshaping input image
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert to float and normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# Convert 1-dimensional class arrays to one-hot encodings (class matrices)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout

# define layers
model = Sequential()
model.add(Input(shape=(28,28,1)))
model.add(Flatten(name='flatten'))
model.add(Dense(400,activation='relu',name='fc_h1'))
model.add(Dense(200,activation='relu',name='fc_h2'))
# model.add(Dense(64,activation='relu',name='fc_h'))
model.add(Dense(10,activation='softmax',name='fc_output'))

# see the network architecture and number of parameters
model.summary()

# finalize the model/network by compiling with the loss, optimizer, and metrics of your choice
# loss types: 'mean_absolute_error', 'mean_squared_error', 'binary_crossentropy', 
#   'categorical_crossentropy', etc.
# optimizer options: SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, etc.
# metrics: for regression - 'mae', 'mse', 'rmse', ...; 
#   for classification - 'accuracy', 'precision', 'recall', ...
# You can use either name or class object(s). Also possible define your own class
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              metrics=['accuracy', tf.keras.metrics.Precision()])

**Train the model**

Train the model by fitting the training data into the model.
def plot_curves(hist):
  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  loss = hist.history['loss']
  val_loss = hist.history['val_loss']

  epochs = range(1, len(acc) + 1)

  # accuracy plots
  plt.plot(epochs, acc, 'g', label='Training acc')
  plt.plot(epochs, val_acc, 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc='upper left')

  # los plots
  plt.figure()
  plt.plot(epochs, loss, 'g', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc='upper right')

  plt.show()

plot_curves(hist)score = model.evaluate(X_test, y_test, verbose=0)

# score contains the metrics passed in the compile method. Here: loss, accuracy, and precision
print(model.metrics_names)
print(score)
import numpy as np

prob = model.predict(X_test)
pred = np.argmax(prob, axis = 1)
label = np.argmax(y_test,axis = 1) 

correct_indices = np.flatnonzero(pred == label) 
incorrect_indices = np.flatnonzero(pred != label) 
print(len(correct_indices)," classified correctly") 
print(len(incorrect_indices)," classified incorrectly")
import itertools
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

conf_mat = tf.math.confusion_matrix(labels=label, predictions=pred).numpy()
plot_confusion_matrix(conf_mat,[0,1,2,3,4,5,6,7,8,9])

## display randomly selected images with top three probabilities
n_images_toshow = 20
n_probs_toshow = 3

# show correctly classified images
imgs, labels = get_random_images(X_test, correct_indices, n_images_toshow,
                                  prob,n_probs_toshow)
show_images(imgs, figsize=(20,15), rows=4, titles=labels,
            suptitle="Sample example of correctly classified images")

# show incorrectly classified images
imgs, labels = get_random_images(X_test, incorrect_indices, n_images_toshow,
                                 prob ,n_probs_toshow)
show_images(imgs, figsize=(20,15), rows=4, titles=labels,
            suptitle="Sample example of incorrectly classified images")



**Evaluate the model on test data**

To check whether the model is best fit for the given problem and corresponding data.

