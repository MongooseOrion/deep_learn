import tensorflow as tf
import numpy as np
import cv2

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

mode = 1    # 0: 训练模型，1: 加载模型进行预测

if mode == 0:
    # 定义模型结构
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    # 编译模型
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    # 训练模型
    model.fit(x_train, y_train, epochs=5)

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])

    # 保存模型
    probability_model.save('./tensorflow_learn/model/mnist_pic_classify.h5')

elif mode == 1:
    # 加载已保存的模型
    loaded_model = tf.keras.models.load_model('./tensorflow_learn/model/mnist_pic_classify.h5')

    # 手动编译加载的模型
    loaded_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

    # 使用加载的模型进行预测
    test_data = x_test[17:40]
    print(f'actual_labels is {y_test[17:40].tolist()}')
    predictions = loaded_model.predict(test_data)

    # 按顺序依次打印预测结果
    predicted_label_mat = []
    for i, prediction in enumerate(predictions):
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label_mat.append(predicted_label)
        #print(f"Image {i}: Predicted label is {predicted_label}, confidence is {confidence:.2f}")
    
    print(f'predic_labels is {predicted_label_mat}')

    # 显示图片
    #cv2.imshow('Image', x_test[2])
    #cv2.waitKey(0)