from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
train_dir = image_path + "train"
validation_dir = image_path + "val"

# create direction for saving weights
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 10

# data generator with data augmentation
# 使用ImageDataGenerator这个类对图像载入并进行预处理，该类中有很多预处理的方法，而且可以自定义处理函数
train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
# 获取训练集的数据数量
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
total_val = val_data_gen.n

# sample_training_images, sample_training_labels = next(train_data_gen)  # label is one-hot coding
#
#
# # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
#
# plotImages(sample_training_images[:5])

model = AlexNet_v1(im_height=im_height, im_width=im_width, class_num=5)
# model = AlexNet_v2(class_num=5)
# model.build((batch_size, 224, 224, 3))  
# # when using subclass model，需要对模型进行build，因为实例化类没有真正实例化模型，必须通过build函数告诉模型batch_input_shape，然后模型才真正创建，在hands on书中没有提到这一点
model.summary()

# using keras high level api for training，很多事情keras自动帮我们处理了，比如dropout在训练的时候是开启的，在验证的时候是关闭的
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 因为在模型中对输出使用了softmax处理，所以设置from_logits为False
              metrics=["accuracy"])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',  # 训练过程中保存模型的设置，h5格式是keras格式，ckpt是tensorflow的格式
                                                save_best_only=True,
                                                save_weights_only=True,  # 只保存了参数，载入的时候需要先创建模型，也可以保存模型和参数，从而可以直接载入带权重的模型
                                                monitor='val_loss')]

# tensorflow2.1 recommend to using fit，之前的tensorflow版本都用fit_generator函数
history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    callbacks=callbacks)

"""
history = model.fit_generator(generator=train_data_gen,
                              steps_per_epoch=total_train // batch_size,
                              epochs=epochs,
                              validation_data=val_data_gen,
                              validation_steps=total_val // batch_size,
                              callbacks=callbacks)
"""

# plot loss and accuracy image
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

# figure 1
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# figure 2
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


# # using keras low level api for training
# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
#
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
#
#
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss = loss_object(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
#
# @tf.function
# def test_step(images, labels):
#     predictions = model(images, training=False)
#     t_loss = loss_object(labels, predictions)
#
#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
#
#
# best_test_loss = float('inf')
# for epoch in range(1, epochs+1):
#     train_loss.reset_states()        # clear history info
#     train_accuracy.reset_states()    # clear history info
#     test_loss.reset_states()         # clear history info
#     test_accuracy.reset_states()     # clear history info
#     for step in range(total_train // batch_size):
#         images, labels = next(train_data_gen)
#         train_step(images, labels)
#
#     for step in range(total_val // batch_size):
#         test_images, test_labels = next(val_data_gen)
#         test_step(test_images, test_labels)
#
#     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#     print(template.format(epoch,
#                           train_loss.result(),
#                           train_accuracy.result() * 100,
#                           test_loss.result(),
#                           test_accuracy.result() * 100))
#     if test_loss.result() < best_test_loss:
#        best_test_loss = test_loss.result()
#        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')
