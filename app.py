from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

from decisionTree import health

app = Flask(__name__)
CORS(app, 
     origins=['http://127.0.0.1:5500'],  
     methods=['GET', 'POST', 'PUT', 'DELETE'],  
     allow_headers=['Content-Type', 'Authorization', 'Accept'],  
     expose_headers=['Content-Type', 'X-Custom-Header'])

# 加载训练好的模型
model = joblib.load('call_predict_model.pkl')
# 加载拟合后的 scaler
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("接收到的数据：", data)
        # 对心率进行校验和处理，将 np.float_ 相关判断替换为 np.float64
        heart_rate = data.get('heart_rate')
        if heart_rate == "" or not isinstance(heart_rate, (float, int, np.float64, np.int_)):
            heart_rate = 0.0
        else:
            heart_rate = float(heart_rate)

        # 对收缩压进行校验和处理，同样替换 np.float_
        systolic_bp = data.get('systolic_bp')
        if systolic_bp == "" or not isinstance(systolic_bp, (float, int, np.float64, np.int_)):
            systolic_bp = 0.0
        else:
            systolic_bp = float(systolic_bp)

        # 对舒张压进行校验和处理，替换 np.float_
        diastolic_bp = data.get('diastolic_bp')
        if diastolic_bp == "" or not isinstance(diastolic_bp, (float, int, np.float64, np.int_)):
            diastolic_bp = 0.0
        else:
            diastolic_bp = float(diastolic_bp)

        # 对通话时段进行校验和处理，替换 np.float_
        call_time = data.get('call_time')
        if call_time == "" or not isinstance(call_time, (float, int, np.float64, np.int_)):
            call_time = 0.0
        else:
            call_time = float(call_time)

        # 对通话时长进行校验和处理，替换 np.float
        call_duration = data.get('call_duration')
        if call_duration == "" or not isinstance(call_duration, (float, int, np.float64, np.int_)):
            call_duration = 0.0
        else:
            call_duration = float(call_duration)
        # 健康状态计算
        health_status = health(heart_rate, systolic_bp, diastolic_bp)
        print("健康状态:", health_status)

        # 组合特征并预测
        X = np.array([[call_duration, health_status, call_time]])
        print("输入特征:", X)
        X_scaled = scaler.transform(X)
        print("标准化特征:", X_scaled)

        # 返回预测结果
        predicted = model.predict(X_scaled)[0]
        # 对推荐联系人进行转换
        recommended_contact = ""
        if predicted[0] == 1:
            recommended_contact = "医生";
        elif predicted[0] == 2:
            recommended_contact = "家人";
        elif predicted[0] == 3:
            recommended_contact = "朋友";
        else:
            recommended_contact = str(predicted[0])
        # 对推荐时间进行转换
        recommended_time = ""
        if predicted[1] == 1:
            recommended_time = "立刻";
        elif predicted[1] == 2:
            recommended_time = "上午";
        elif predicted[1] == 3:
            recommended_time = "下午";
        elif predicted[1] == 4:
            recommended_time = "晚上";
        else:
            recommended_time = str(predicted[1])
        print("预测结果:", predicted)

        # 返回结果
        return jsonify({
            "recommended_contact": recommended_contact,
            "recommended_time": recommended_time
        })
    except Exception as e:
        print("发生错误：", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5005)
    