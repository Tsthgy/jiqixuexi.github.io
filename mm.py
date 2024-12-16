import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from datetime import datetime

# 加载CSV文件
df = pd.read_csv('activities.csv')

# 将时间转换为数值型数据（从1970年1月1日以来的分钟数）
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp_minutes'] = (df['timestamp'] - pd.Timestamp('1970-01-01')).dt.total_seconds() / 60

# 将事件类型转换为数值型数据
le = LabelEncoder()
df['event_encoded'] = le.fit_transform(df['event'])

# 创建一个新的列来表示下一次事件发生的时间间隔
# 首先，对事件进行排序
df.sort_values('timestamp', inplace=True)

# 然后，对于每个事件，计算与下一个相同类型事件的时间差
df['time_to_next'] = df.groupby('event_encoded')['timestamp_minutes'].transform(lambda x: x.diff().shift(-1))

# 填充 NaN 值，这里使用均值填充
df['time_to_next'].fillna(df['time_to_next'].mean(), inplace=True)

# 选择特征和标签
X = df[['timestamp_minutes', 'event_encoded']]
y = df['time_to_next']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 保存模型到文件
joblib.dump(model, 'linear_regression_model.joblib')

print("Model trained and saved successfully.")