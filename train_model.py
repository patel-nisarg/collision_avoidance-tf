# This script uses transfer learning with MobileNetV2 to detect blocked versus free paths. At least 100 images of
# each should be provided. Examples of good "blocked" and "free" path images are provided in ReadMe.


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224
image_channel = 3

pre_trained_model = MobileNetV2(input_shape=(image_size, image_size, image_channel),
                                weights='imagenet',
                                include_top=False)
# have to specify include_top = False or else input shape must be (299, 299, 3). Also choose weights or default to
# ImageNet training weights. Choosing "None" which will require either loading weights via
# model.load_weights(filename) or training from random initialization (not recommended)!

for layer in pre_trained_model.layers:
    layer.trainable = False

last_output = pre_trained_model.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

filename = "F:\\Documents\\Jetbot\\collision_avoidance_tf\\output\\"  # Points to dataset of blocked and free images.

train_aug = ImageDataGenerator(rotation_range=0.2,
                               shear_range=0.1,
                               zoom_range=0.1,
                               fill_mode='nearest',
                               horizontal_flip=True,
                               rescale=1.0 / 255)

val_aug = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_aug.flow_from_directory(directory=filename + "train",
                                          target_size=(image_size, image_size),
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=8,
                                          shuffle=True,
                                          seed=123)

val_gen = val_aug.flow_from_directory(directory=filename + "val",
                                      target_size=(image_size, image_size),
                                      color_mode='rgb',
                                      class_mode='binary',
                                      batch_size=8,
                                      shuffle=True,
                                      seed=123)

model = tf.keras.Model(pre_trained_model.input, x)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
save_path = 'F:\\Documents\\Jetbot\\collision_avoidance_tf\\model_save'
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
num_epochs = 50
history = model.fit(train_gen, epochs=30, validation_data=val_gen, callbacks=[checkpoint])
