import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.models.load_model('./tensorflow_learn/model/mnist_pic_classify.h5')
# 手动编译加载的模型
model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

spec = (tf.TensorSpec((1, 28, 28), tf.float32, name="input"),)
output_path = './tensorflow_learn/model/mnist_pic_classify.onnx'
 
# 转换并保存onnx模型，opset决定选用的算子集合
tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

# 删除多余的算子
original_model = onnx.load("./tensorflow_learn/model/mnist_pic_classify.onnx")
del original_model.opset_import[1]
onnx.save(original_model, './tensorflow_learn/model/mnist_pic_classify_mod.onnx')