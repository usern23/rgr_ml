import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model_save_path = 'streamlit/streamlit-models/'

airlines_data = pd.read_csv("streamlit/airlines.csv")
X = airlines_data.drop('Delay', axis=1)
y = airlines_data['Delay']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f1f3f6;
}
h1 {
    color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

# Сайдбар для навигации
st.sidebar.image("streamlit/erlan.jpg", width=100)
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ("Информация о разработчике", "Информация о наборе данных", "Визуализации данных", "Предсказание модели ML")
)


# Функции для каждой страницы
def page_developer_info():
    st.title("Информация о разработчике")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Контактная информация")
        st.write("ФИО: Чуакбаев Ерлан Арманович")
        st.write("Номер учебной группы: ФИТ-222")
    
    with col2:
        st.header("Фотография")
        st.image("streamlit/erlan.jpg", width=200)
    
    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

def page_dataset_info():
    st.title("Информация о наборе данных")

    st.markdown("""
    ## Описание Датасета Airlines
    **Файл датасета:** `airlines.csv`

    **Описание:**
    Данный датасет содержит статистическую информацию о задержках рейсов авиалиний. Включает следующие столбцы:

    - `Flight`: Индекс записи.
    - `DayOfWeek`: Время до конца раунда.
    - `Time`: Счёт команды контр-террористов.
    - `Length`: Счёт команды террористов.
    - `Delay`: Общее здоровье команды контр-террористов.
                
    **Особенности предобработки данных:**
    - Удаление лишних столбцов, например, 'index'.
    - Обработка пропущенных значений.
    - Нормализация числовых данных для улучшения производительности моделей.
    - Кодирование категориальных переменных.
    """)

def page_data_visualization():
    st.title("Визуализации данных Airlines")

    # Визуализация 1: Гистограмма задержек рейсов
    plt.figure(figsize=(10, 6))
    plt.hist(airlines_data['Delay'], bins=10, color='blue', alpha=0.7)
    plt.title('Распределение задержек рейсов')
    plt.xlabel('Задержка')
    plt.ylabel('Количество рейсов')
    st.pyplot(plt)  # Отображение графика в Streamlit

    # Визуализация 2: Точечная диаграмма для Length и Delay
    plt.figure(figsize=(10, 6))
    plt.scatter(airlines_data['Length'], airlines_data['Delay'], color='green')
    plt.title('Взаимосвязь между продолжительностью полёта и задержкой')
    plt.xlabel('Продолжительность полёта')
    plt.ylabel('Задержка')
    st.pyplot(plt) 

    # Визуализация 3: График количества рейсов по дням недели
    plt.figure(figsize=(10, 6))
    airlines_data['DayOfWeek'] = airlines_data['DayOfWeek'].round(2)
    airlines_data['DayOfWeek'].value_counts().plot(kind='bar', color='orange')
    plt.title('Количество рейсов по дням недели')
    plt.xlabel('День недели')
    plt.ylabel('Количество рейсов')
    st.pyplot(plt) 

    # Визуализация 4: Гистограмма для различных авиакомпаний
    plt.figure(figsize=(10, 6))
    airlines_data['Flight'] = airlines_data['Flight'].round(2)
    airlines_data['Flight'][:30].value_counts().plot(kind='bar', color='purple')
    plt.title('Количество рейсов по продолжительности полета')
    plt.xlabel('Длительность полета')
    plt.ylabel('Количество рейсов')
    st.pyplot(plt) 

    # Визуализация 4: Гистограмма для различных авиакомпаний
    plt.figure(figsize=(10, 6))
    airlines_data['Time'] = airlines_data['Time'].round(2)
    airlines_data['Time'][:20].value_counts().plot(kind='bar', color='purple')
    plt.title('Time')
    plt.xlabel('Время задержки')
    plt.ylabel('Количество рейсов')
    st.pyplot(plt) 
    

# Функция для загрузки моделей
def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

def page_ml_prediction():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")
        airlines_data = pd.read_csv('streamlit/airlines.csv')

        # Интерактивные поля для ввода данных
        input_data = {}
        all_columns = airlines_data.columns.tolist()
        feature_names = all_columns
        feature_names.remove("Delay") 
        for feature in feature_names:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=100000.0, value=50.0)

        if st.button('Сделать предсказание'):
            # Загрузка моделей
            model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2 = load_models()

            input_df = pd.DataFrame([input_data])
            
            st.write("Входные данные:", input_df)

            # Используем масштабировщик, обученный на обучающих данных
            scaler = StandardScaler().fit(X_train)
            scaled_input = scaler.transform(input_df)

            # Делаем предсказания
            prediction_ml1 = model_ml1.predict(scaled_input)
            prediction_ml3 = model_ml3.predict(scaled_input)
            prediction_ml4 = model_ml4.predict(scaled_input)
            prediction_ml5 = model_ml5.predict(scaled_input)
            prediction_ml6 = (model_ml6.predict(scaled_input) > 0.5).astype(int)

            # Вывод результатов
            st.success(f"Результат предсказания LogisticRegression: {prediction_ml1[0]}")
            st.success(f"Результат предсказания XGBClassifier: {prediction_ml3[0]}")
            st.success(f"Результат предсказания BaggingClassifier: {prediction_ml4[0]}")
            st.success(f"Результат предсказания StackingClassifier: {prediction_ml5[0]}")
            st.success(f"Результат предсказания нейронной сети Tensorflow: {prediction_ml6[0]}")
    else:
        try:
            model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
            model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
            model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
            model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
            model_ml3 = XGBClassifier()
            model_ml3.load_model(model_save_path + 'model_ml3.json')
            model_ml6 = load_model(model_save_path + 'model_ml6.h5')

            # Сделать предсказания на тестовых данных
            cluster_labels = model_ml2.predict(X_test)
            predictions_ml1 = model_ml1.predict(X_test)
            predictions_ml4 = model_ml4.predict(X_test)
            predictions_ml5 = model_ml5.predict(X_test)
            predictions_ml3 = model_ml3.predict(X_test)
            predictions_ml6 = model_ml6.predict(X_test).round() # Округление для нейронной сети

            # Оценить результаты
            rand_score_ml2 = rand_score(y_test, cluster_labels)
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"rand_score KMeans: {rand_score_ml2}")
            st.success(f"Точность LogisticRegression: {accuracy_ml1}")
            st.success(f"Точность XGBClassifier: {accuracy_ml4}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml5}")
            st.success(f"Точность StackingClassifier: {accuracy_ml3}")
            st.success(f"Точность нейронной сети Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")


if page == "Информация о разработчике":
    page_developer_info()
elif page == "Информация о наборе данных":
    page_dataset_info()
elif page == "Визуализации данных":
    page_data_visualization()
elif page == "Предсказание модели ML":
    page_ml_prediction()
