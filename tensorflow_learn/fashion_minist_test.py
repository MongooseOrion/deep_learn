import tensorflow as tf
import numpy as np

# 加载 Fashion MNIST 数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 归一化处理
train_images, test_images = train_images / 255.0, test_images / 255.0

mode = 1    # 0: 训练模型，1: 加载模型进行预测

if mode == 0:

    # 定义模型结构
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # 编译模型
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=10)

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f'test_loss: {test_loss}, test_acc: {test_acc}')

    # 保存模型
    probability_model.save('./tensorflow_learn/model/fashion_mnist_pic_classify.h5')

elif mode == 1:
    # 加载模型
    model = tf.keras.models.load_model('./tensorflow_learn/model/fashion_mnist_pic_classify.h5')

    # 预测
    predictions = model.predict(test_images[15:25])

    # 取出预测结果
    for i, prediction in enumerate(predictions):
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        print(f'predicted_label: {class_names[predicted_label]}, \ttrue_label: {class_names[test_labels[i]]}, \tconfidence: {confidence:.2f}')