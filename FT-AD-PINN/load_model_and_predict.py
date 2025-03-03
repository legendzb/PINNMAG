import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import griddata


# 定义损失函数
def loss_fn(model, inputs, targets):
    B_pred = model(inputs)  # 预测地磁场分量X,Y,Z,F
    data_loss = tf.sqrt(tf.reduce_mean(tf.square(B_pred - targets)))  # 数据损失 (RMSE)
    return data_loss

# 加载模型
model = tf.keras.models.load_model('pinn_model', custom_objects={'loss_fn': loss_fn})

# 从文件中读取数据
data = pd.read_csv('data2.csv')

# 提取输入特征
latitude_raw = data['Latitude'].values
longitude_raw = data['Longitude'].values
altitude_raw = data['Radius'].values
time_raw = data['Timestamp'].values

latitude_raw = 90 - latitude_raw  # 将纬度转换为余纬度

# 提取真实的磁场分量
true_magnetic_field = data[['北向', '东向', '垂向', 'F']].values

# 加载均值和标准差
data_stats = np.load('data_stats.npz')
longitude_mean = data_stats['longitude_mean']
latitude_mean = data_stats['latitude_mean']
altitude_mean = data_stats['altitude_mean']
time_mean = data_stats['time_mean']
longitude_std = data_stats['longitude_std']
latitude_std = data_stats['latitude_std']
altitude_std = data_stats['altitude_std']
time_std = data_stats['time_std']

# 标准化输入数据
latitude = (latitude_raw - latitude_mean) / latitude_std
longitude = (longitude_raw - longitude_mean) / longitude_std
altitude = (altitude_raw - altitude_mean) / altitude_std
time = (time_raw - time_mean) / time_std

# 准备输入数据
inputs = np.stack([latitude, longitude, altitude, time], axis=1)

# 转换为 TensorFlow 张量
inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)

# 调用模型进行预测
predicted_magnetic_field = model(inputs_tensor).numpy()


# 计算均方根误差 (RMSE)
rmse_x = np.sqrt(np.mean((predicted_magnetic_field[:, 0] - true_magnetic_field[:, 0]) ** 2))
rmse_y = np.sqrt(np.mean((predicted_magnetic_field[:, 1] - true_magnetic_field[:, 1]) ** 2))
rmse_z = np.sqrt(np.mean((predicted_magnetic_field[:, 2] - true_magnetic_field[:, 2]) ** 2))
rmse_f = np.sqrt(np.mean((predicted_magnetic_field[:, 3] - true_magnetic_field[:, 3]) ** 2))

print(f'RMSE for X component: {rmse_x}')
print(f'RMSE for Y component: {rmse_y}')
print(f'RMSE for Z component: {rmse_z}')
print(f'RMSE for F component: {rmse_f}')

# 可视化地磁分量的残差
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(predicted_magnetic_field[:, 0] - true_magnetic_field[:, 0], label='Residual Error')
plt.legend(loc='upper right')
plt.title('X Component(nT)')

plt.subplot(2, 2, 2)
plt.plot(predicted_magnetic_field[:, 1] - true_magnetic_field[:, 1], label='Residual Error')
plt.legend(loc='upper right')
plt.title('Y Component(nT)')

plt.subplot(2, 2, 3)
plt.plot(predicted_magnetic_field[:, 2] - true_magnetic_field[:, 2], label='Residual Error')
plt.legend(loc='upper right')
plt.title('Z Component(nT)')

plt.subplot(2, 2, 4)
plt.plot(predicted_magnetic_field[:, 3] - true_magnetic_field[:, 3], label='Residual Error')
plt.legend(loc='upper right')
plt.title('F Component(nT)')

plt.tight_layout()
plt.show()

np.savetxt('predictions.csv', predicted_magnetic_field, delimiter=',', header='X, Y, Z, F', comments='')
