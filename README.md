
# Лабораторная работа по программной инженерии.

## Задачи и используемый стек

### 1. Определение спама в тексте
- Модель: RuBERT-tiny2
- Стек: Python, Hugging Face Transformers, PyTorch

### 2. Распознавание окружающих звуков
- Модель: YAMNet
- Стек: Python, TensorFlow, TensorFlow Hub, NumPy, SciPy

### 3. Распознавание болезней растений
- Модель: MobileNetV2
- Стек: Python, Hugging Face Transformers, PyTorch, Pillow

### 4. Подсчёт автомобилей на видео
- Модель: YOLO11n
- Стек: Python, Ultralytics, PyTorch, OpenCV

### 5. API и тестирование
- Стек: FastAPI, Pydantic, Uvicorn, pytest, HTTPX
- API: `main.py`
- Тесты: `test_main.py`
- CI: GitHub Actions

## Установка

Рекомендуется использовать Python 3.12.

Создать виртуальное окружение:

    python -m venv venv

Активировать окружение (Windows):

    venv\Scripts\activate

Установить зависимости:

    python -m pip install -r requirements.txt

## Запуск задач

Все команды выполняются из корневой папки проекта.

### Задача 1

    python task1_text/spam_detector.py

### Задача 2

Поместить WAV-файл `miaow_16k.wav` в папку `task2_audio/samples`.

    python task2_audio/sound_detector.py

### Задача 3

Поместить изображение `tomato_leaf.jpg` в папку `task3_image/samples`.

    python task3_image/plant_detector.py

### Задача 4

Поместить видео `traffic.mp4` в папку `task4_video/samples`.

    python task4_video/car_counter.py

Результат сохраняется в `task4_video/results`.

## Запуск API

    python -m uvicorn main:app --reload

Документация API:

http://127.0.0.1:8000/docs

Для определения спама используется POST-запрос `/predict`.

## Запуск тестов

Установить зависимости для тестов:

    python -m pip install -r requirements-test.txt

Запустить тесты:

    python -m pytest test_main.py -v

Автоматический запуск тестов настроен через GitHub Actions.
