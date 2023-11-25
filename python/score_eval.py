# 自定义建立了一个线性回归模型用于预测成绩

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 从文本文件加载数据
with open('geo_data.txt', 'r') as file:
    lines = file.readlines()

# 解析数据
data = [list(map(float, line.split(','))) for line in lines]

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# 转换为NumPy数组
train_data = np.array(train_data)
test_data = np.array(test_data)

# 标准化数据
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = torch.tensor(scaler_x.fit_transform(train_data[:, 0].reshape(-1, 1)), dtype=torch.float32)
Y_train = torch.tensor(scaler_y.fit_transform(train_data[:, 1].reshape(-1, 1)), dtype=torch.float32)

X_test = torch.tensor(scaler_x.transform(test_data[:, 0].reshape(-1, 1)), dtype=torch.float32)
Y_test = torch.tensor(scaler_y.transform(test_data[:, 1].reshape(-1, 1)), dtype=torch.float32)

# 定义神经网络模型
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    Z_pred = model(X_train)
    loss = criterion(Z_pred, Y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    Z_pred_test = model(X_test)

# 反标准化预测结果
T_pred_test = scaler_y.inverse_transform(Z_pred_test)

# 输出测试结果
for i in range(len(test_data)):
    print(f'Original: {test_data[i, 0]}, Transformed (actual): {test_data[i, 1]}, Transformed (predicted): {T_pred_test[i, 0]:.2f}')

# 保存模型
torch.save(model.state_dict(), 'your_model.pth')

# 创建一个新的模型实例
loaded_model = RegressionModel()

# 加载保存的状态字典
loaded_model.load_state_dict(torch.load('your_model.pth'))

# 将模型设置为评估模式
loaded_model.eval()

# 从用户输入获取新数据
try:
    raw_input_score = float(input('Enter the new raw score: '))
except ValueError:
    print('无效的输入，请输入一个有效的数字。')
    exit()

# 使用相同的标准化器标准化新输入
new_data = torch.tensor(scaler_x.transform([[raw_input_score]]), dtype=torch.float32)

# 使用加载的模型进行预测
with torch.no_grad():
    Z_pred = loaded_model(new_data)

# 反标准化以获得最终的预测结果
T_pred = scaler_y.inverse_transform(Z_pred)

# 输出预测结果
print(f'Original: {raw_input_score}, Transformed (predicted): {T_pred.item():.2f}')