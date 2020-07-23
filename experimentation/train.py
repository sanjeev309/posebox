import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


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
data = np.load(os.path.join(OUTPUT,'data.npy'))
target = np.load(os.path.join(OUTPUT,'target.npy'))
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
    base_model = keras.applications.MobileNetV2(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    print(base_model.summary())

    model = keras.Sequential([

        base_model,
        layers.Flatten(),
        layers.Dense(64),
        layers.Dense(16),
        layers.Dense(num_output, activation="sigmoid"),

    ])
    optimizer = keras.optimizers.RMSprop(0.01)

    model.compile(loss=keras.losses.mean_absolute_error,
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
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    return build_model()


model = make_or_restore_model()
callbacks = [
    # This callback saves a SavedModel every 5 epochs.
    # We include the training loss in the folder name.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
        # save_freq=4)
        period=5)
]

model.fit(data, target, verbose=1, epochs=50, batch_size=batch_size, validation_split= 0.2, callbacks=callbacks)
