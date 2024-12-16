from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten  # type: ignore
from sklearn.model_selection import train_test_split
import time

app = Flask(__name__)
# 跨域配置，假设前端运行在本地5500端口
CORS(app, 
     origins=['http://127.0.0.1:5500'],  
     methods=['GET', 'POST', 'PUT', 'DELETE'],  
     allow_headers=['Content-Type', 'Authorization', 'Accept'],  
     expose_headers=['Content-Type', 'X-Custom-Header'])

# 用于控制是否重新获取传感器数据的全局变量，初始化为True表示可以正常获取数据
should_update_data = True  
# 模拟获取传感器数据的函数（实际应用中需替换为真实传感器数据读取）
def get_sensor_data():
    global should_update_data
    if should_update_data:
        # 这里简单模拟生成数据，实际应从传感器读取
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(9.5, 10.5)
        return np.array([[x, y, z]])
    return None  # 如果不允许更新数据，返回None

# 假设这里已经有训练好的模型trained_model，在实际应用中应确保模型已正确加载
trained_model = tf.keras.models.load_model('fall_detection_model.keras')
# 检查模型是否已经编译，如果未编译再进行编译（此处通过查看optimizer属性判断，若为None表示未编译）
if trained_model.optimizer is None:
    trained_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 摔倒检测函数
def detect_fall():
    # 获取传感器数据
    sensor_data = get_sensor_data()
    if sensor_data is not None:
        # 预处理数据（这里假设预处理逻辑与训练时相同，若训练时有其他预处理比如归一化等需补充完整）
        sequence_length = 1
        num_features = 3
        sensor_data = sensor_data.reshape(1, sequence_length, num_features)
        # 进行预测
        prediction = trained_model.predict(
            sensor_data)
        # 根据预测结果判断是否摔倒
        return prediction[0][0] > 0.5
    return None  # 如果没有获取到新数据，返回None

@app.route('/detect_fall', methods=['GET'])
def check_fall():
    result = detect_fall()
    message = "老人摔倒，请求紧急援助！" if result else "目前未检测到摔倒情况"
    return jsonify({"result": message})

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global should_update_data
    should_update_data = True
    return jsonify({"message": "数据已重置，可重新检测"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)