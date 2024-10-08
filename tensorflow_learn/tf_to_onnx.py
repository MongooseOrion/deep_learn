''' ======================================================================
* Copyright (c) 2023, MongooseOrion.
* All rights reserved.
*
* The following code snippet may contain portions that are derived from
* OPEN-SOURCE communities, and these portions will be licensed with: 
*
* <NULL>
*
* If there is no OPEN-SOURCE licenses are listed, it indicates none of
* content in this Code document is sourced from OPEN-SOURCE communities. 
*
* In this case, the document is protected by copyright, and any use of
* all or part of its content by individuals, organizations, or companies
* without authorization is prohibited, unless the project repository
* associated with this document has added relevant OPEN-SOURCE licenses
* by github.com/MongooseOrion. 
*
* Please make sure using the content of this document in accordance with 
* the respective OPEN-SOURCE licenses. 
* 
* THIS CODE IS PROVIDED BY https://github.com/MongooseOrion. 
* FILE ENCODER TYPE: UTF-8
* ========================================================================
'''
# 转换 tensorflow2 模型为 onnx 模型
import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.models.load_model('./tensorflow_learn/model/mnist_pic_classify.h5')
# 手动编译加载的模型
model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

# 指定输入张量的形状
spec = (tf.TensorSpec((1, 28, 28), tf.float32, name="input"),)
output_path = './tensorflow_learn/model/mnist_pic_classify.onnx'
 
# 转换并保存onnx模型，opset决定选用的算子集合
tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

# 删除模型中多余的算子类型（如果有），否则会导致 CANN ATC 转换失败
original_model = onnx.load("./tensorflow_learn/model/mnist_pic_classify.onnx")
del original_model.opset_import[1]
onnx.save(original_model, './tensorflow_learn/model/mnist_pic_classify_mod.onnx')