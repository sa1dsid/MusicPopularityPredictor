# Импорты
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# === 1. Загрузка и подготовка данных ===
# Загрузка датасета
data = pd.read_csv('data.csv')

# Выбор признаков и целевой переменной
X = data[['instrumentalness', 'year', 'acousticness', 'energy', 'loudness']]  # Признаки
y = data['popularity']  # Целевая переменная

# Нормализация данных
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Нормализация признаков
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))  # Нормализация целевой переменной

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === 2. Создание нейросети ===
model = Sequential([
    Dense(64, input_dim=5, activation='relu'),  # Входной слой: 5 признаков
    Dense(32, activation='relu'),  # Скрытый слой
    Dense(16, activation='relu'),  # Скрытый слой
    Dense(1, activation='linear')  # Выходной слой: предсказание popularity
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === 3. Обучение модели ===
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# === 4. Оценка модели ===
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error on Test Data: {mae:.4f}')

# === 5. Предсказание ===
predictions = model.predict(X_test)
print("Примеры предсказаний:", predictions[:5])