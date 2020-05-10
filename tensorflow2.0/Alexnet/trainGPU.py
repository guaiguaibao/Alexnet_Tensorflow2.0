import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os
import time
import glob  # 使用glob来对目录以及图像文件名称进行模糊查找
import random

# 通过以下两行代码来告诉tensorflow只能发现设备的第一块gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 根据模型大小灵活占用gpu的内存，而不是死板的将gpu内存都占满
    except RuntimeError as e:
        print(e)
        exit(-1)

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

# class dict
data_class = [cla for cla in os.listdir(train_dir) if ".txt" not in cla]
# 这里因为没有用到ImageDataGenerator类来加载图像，所以图像分类标签要遍历train目录来获取
class_num = len(data_class)
class_dict = dict((value, index) for index, value in enumerate(data_class))

# reverse value and key of dict
inverse_dict = dict((val, key) for key, val in class_dict.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# load train images list -- 这里不是直接操作图像，而是操作图像的路径
train_image_list = glob.glob(train_dir+"/*/*.jpg")
random.shuffle(train_image_list)
train_num = len(train_image_list)
train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]
# 在linux中os.path.sep就是/，通过图像所在目录名称找到图像对应的label 0 1 2 3 4

# load validation images list
val_image_list = glob.glob(validation_dir+"/*/*.jpg")
random.shuffle(val_image_list)
val_num = len(val_image_list)
val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]


# 在这里没有将图像进行归一化和左右翻转操作
def process_path(img_path, label):
    label = tf.one_hot(label, depth=class_num)  # 对label进行one hot编码
    image = tf.io.read_file(img_path)  # 根据每个图像的路径，将图像载入，并且对其进行一些处理
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [im_height, im_width])
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
# AUTOTUNE to make TensorFlow choose the right number of threads dynamically based on the available CPU，也就是tf来根据实际情况决定开几个线程

# load train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
train_dataset = train_dataset.shuffle(buffer_size=train_num)\
                             .map(process_path, num_parallel_calls=AUTOTUNE)\
                             .repeat().batch(batch_size).prefetch(AUTOTUNE)
# 利用map函数对train_dataset中的每个元素进行处理
# 使用tf.data.Dataset.from_tensor_slices是可以使用多线程的，但是ImageDataGenerator是不会使用多线程的，由于在这里使用了GPU训练网络，为了缩短图像载入和预处理的时间，推荐使用tf.data方式多线程进行

# load train dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                         .repeat().batch(batch_size)

# 实例化模型
model = AlexNet_v1(im_height=im_height, im_width=im_width, class_num=5)
# model = AlexNet_v2(class_num=5)
# model.build((batch_size, 224, 224, 3))  # when using subclass model
model.summary()

# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


best_test_loss = float('inf')
train_step_num = train_num // batch_size
val_step_num = val_num // batch_size
for epoch in range(1, epochs+1):
    train_loss.reset_states()        # clear history info
    train_accuracy.reset_states()    # clear history info
    test_loss.reset_states()         # clear history info
    test_accuracy.reset_states()     # clear history info

    t1 = time.perf_counter()
    for index, (images, labels) in enumerate(train_dataset):
        train_step(images, labels)
        if index+1 == train_step_num:
            break
    print(time.perf_counter()-t1)

    for index, (images, labels) in enumerate(val_dataset):
        test_step(images, labels)
        if index+1 == val_step_num:
            break

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    if test_loss.result() < best_test_loss:
        model.save_weights("./save_weights/myAlex_{}.ckpt".format(epoch), save_format='tf')

# # using keras high level api for training
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#               metrics=["accuracy"])
#
# callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex_{epoch}.h5',
#                                                 save_best_only=True,
#                                                 save_weights_only=True,
#                                                 monitor='val_loss')]
#
# # tensorflow2.1 recommend to using fit
# history = model.fit(x=train_dataset,
#                     steps_per_epoch=train_num // batch_size,
#                     epochs=epochs,
#                     validation_data=val_dataset,
#                     validation_steps=val_num // batch_size,
#                     callbacks=callbacks)
