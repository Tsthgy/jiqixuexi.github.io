import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten # type: ignore
from sklearn.model_selection import train_test_split


# 定义创建传感器数据集的函数
def create_sensor_dataset(num_normal_samples, num_fall_samples):
    # 模拟生成正常行为数据
    def generate_normal_behavior_data(num_samples):
        data = []
        # 模拟缓慢行走动作数据等（这里简化模拟，你可按需完善）
        for _ in range(num_samples):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(0.1, 0.3)
            z = np.random.uniform(9.5, 10.5)
            data.append([x, y, z])
        print(np.array(data).shape)  # 打印生成的正常行为数据维度
        print("正常行为数据元素数量:", np.array(data).size)  # 新增打印元素数量
        return np.array(data)

    # 模拟生成摔倒行为数据
    def generate_fall_behavior_data(num_samples):
        data = []
        # 模拟摔倒瞬间及之后的数据（简化模拟）
        for _ in range(num_samples):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(9.5, 10.5)
            data.append([x, y, z])
        print(np.array(data).shape)  # 打印生成的摔倒行为数据维度
        print("正常行为数据元素数量:", np.array(data).size)  # 新增打印元素数量
        return np.array(data)

    normal_data = generate_normal_behavior_data(num_normal_samples)
    fall_data = generate_fall_behavior_data(num_fall_samples)

    # 合并数据
    X = np.concatenate((normal_data, fall_data), axis=0)
    print(X.shape)  # 打印合并后的数据维度
    print("合并前数据X元素数量:", X.size)  # 新增打印元素数量
    # 创建标签，正常行为为0，摔倒行为为1
    y = np.concatenate((np.zeros(len(normal_data)), np.ones(len(fall_data))), axis=0)
    return X, y


# 划分数据集
def split_dataset(X, y):
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# 数据预处理
def preprocess_data(X_train, X_val, X_test):
    # 归一化数据，这里简单将每个特征维度归一化到0-1区间
    print("归一化前训练集维度:", X_train.shape)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    print("归一化后训练集维度:", X_train.shape)
    X_val = (X_val - mean) / std
    print("归一化后验证集维度:", X_val.shape)
    X_test = (X_test - mean) / std
    print("归一化后测试集维度:", X_test.shape)

    # 调整数据形状，假设我们将其看作一个时间序列数据（这里简化处理），sequence_length为数据长度，num_features为特征数（比如3个轴向加速度）
    sequence_length = 1
    num_features = 3
    batch_size_train = X_train.shape[0]
    batch_size_val = X_val.shape[0]
    batch_size_test = X_test.shape[0]
    X_train = X_train.reshape(batch_size_train, sequence_length, num_features)
    print("形状调整后训练集维度:", X_train.shape)
    X_val = X_val.reshape(batch_size_val, sequence_length, num_features)
    print("形状调整后验证集维度:", X_val.shape)
    X_test = X_test.reshape(batch_size_test, sequence_length, num_features)
    print("形状调整后测试集维度:", X_test.shape)
    return X_train, X_val, X_test


# 构建CNN模型
def build_cnn_model(sequence_length):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=(sequence_length, 3), padding='SAME'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))  # 添加padding="SAME"
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='SAME'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))  # 添加padding="SAME"
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 训练模型
def train_model(X_train, y_train, X_val, y_val):
    model = build_cnn_model(1)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                        validation_data=(X_val, y_val), verbose=1)
    return model, history


if __name__ == "__main__":
    num_normal_samples = 200  # 正常行为数据数量
    num_fall_samples = 400  # 摔倒行为数据数量

    # 创建传感器数据集
    X, y = create_sensor_dataset(num_normal_samples, num_fall_samples)
    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    # 预处理数据
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    # 训练模型
    trained_model, history = train_model(X_train, y_train, X_val, y_val)

    # 在测试集上评估模型最终性能
    test_loss, test_acc = trained_model.evaluate(X_test, y_test)
    print(f"测试集损失: {test_loss}")
    print(f"测试集准确率: {test_acc}")
    trained_model.save('fall_detection_model.keras')