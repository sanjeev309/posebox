import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


num_output = 8
input_shape = (512, 512, 3)

batch_size = 10

IMAGES_FOLDER = 'resized_frames'
ANNOTATION_FILE = 'annotation_formatted.csv'
OUTPUT = 'output'

checkpoint_dir = OUTPUT + "/ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

### Load data and target
data = np.load(os.path.join(OUTPUT, 'data.npy'))
target = np.load(os.path.join(OUTPUT, 'target.npy'))
num_samples = data.shape[0]
print("num_samples", num_samples)

### Train / Test split
TRAIN_RATIO = 0.8

X_train = data[0: int(num_samples * TRAIN_RATIO) - 1]
y_train = target[0: int(num_samples * TRAIN_RATIO) - 1]

X_test = data[int(num_samples * TRAIN_RATIO): num_samples - 1]
y_test = target[int(num_samples * TRAIN_RATIO): num_samples - 1]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


def build_model():
    base_model = keras.applications.MobileNetV2(input_shape=(127, 127, 3),
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    print(base_model.summary())

    model = keras.Sequential([

        layers.Input(shape=input_shape),
        layers.Conv2D(3, (2, 2), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(3, (2, 2), activation='relu'),
        layers.MaxPool2D((2, 2)),
        base_model,
        layers.Flatten(),
        layers.Dense(64),
        layers.Dense(32),
        layers.Dense(16),
        layers.Dense(num_output, activation="sigmoid"),

    ])
    optimizer = keras.optimizers.SGD(0.01)

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        name = os.path.basename(latest_checkpoint)
        first, last = "weights.", "-"
        start = name.index(first) + len(first)
        end = name.index(last, start)
        initial_epoch = int(name[start:end])
        print(initial_epoch)
        print('Restoring from', latest_checkpoint)
        return initial_epoch, keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    initial_epoch = 0
    return initial_epoch, build_model()


initial_epoch, model = make_or_restore_model()
callbacks = [
    # This callback saves a SavedModel every 5 epochs.
    # We include the training loss in the folder name.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/weights.{epoch:1d}-{val_loss:.4f}.hdf5',
        # save_freq=4)
        period=5)
]

print("Initial Epoch: ", initial_epoch)
history = model.fit(data, target, verbose=1, epochs=5000, batch_size=batch_size, initial_epoch=initial_epoch,
          validation_split=0.2, callbacks=callbacks)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()

plt.savefig(os.path.join(OUTPUT,'loss_accuracy.png'))
plt.savefig(os.path.join(OUTPUT,'loss_accuracy.pdf'))