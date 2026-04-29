"""
Подбор лучшего гиперпараметра alpha для функций активации tanh(alpha*x)
Сравнение Keras и PyTorch реализаций
"""

import os
import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import random

# ============================================================================
# КОНСТАНТЫ
# ============================================================================
IMG_SIZE = 32
ALPHABET = "йклмнопрсту"
NUM_SAMPLES_PER_LETTER = 1000
DATASET_DIR = "dataset_images"
ALPHA = 3
T0 = 0

# Значения alpha для тестирования
ALPHAS_TO_TEST = [1, 2, 3, 4, 5]

# ============================================================================
# ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================================
def get_local_fonts():
    font_paths = [f for f in os.listdir('.') if f.endswith('.ttf')]
    if not font_paths:
        sys_path = "C:\\\\Windows\\\\Fonts\\\\"
        common_fonts = ['arial.ttf', 'times.ttf', 'verdana.ttf']
        font_paths = [os.path.join(sys_path, f) for f in common_fonts if os.path.exists(os.path.join(sys_path, f))]
    if not font_paths:
        raise RuntimeError("Шрифты .ttf не найдены!")
    return font_paths

def generate_char_image(char, font_path, save_path=None):
    canvas_size = IMG_SIZE * 3
    bg_color = random.randint(235, 255)
    img = Image.new('L', (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(int(IMG_SIZE * 0.9), int(IMG_SIZE * 1.3))
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_size - w) / 2 - bbox[0]
    y = (canvas_size - h) / 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)

    img = img.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, fillcolor=bg_color)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 0.8)))

    center = canvas_size // 2
    s = IMG_SIZE // 2
    jx, jy = random.randint(-2, 2), random.randint(-2, 2)
    img = img.crop((center - s + jx, center - s + jy, center + s + jx, center + s + jy))

    if save_path:
        img.save(save_path)

    img = ImageOps.invert(img)
    arr = np.array(img).astype(np.float32) / 255.0
    if random.random() < 0.4:
        arr += np.random.normal(0, 0.01, arr.shape)

    arr = (np.clip(arr, 0, 1) > 0.4).astype(np.float32)
    return arr.flatten()

def generate_dataset():
    fonts = get_local_fonts()
    all_images = []
    all_labels = []

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Создана директория: {DATASET_DIR}")

    print("Генерация данных...")
    for label, char in enumerate(ALPHABET):
        char_dir = os.path.join(DATASET_DIR, char)
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)

        for i in range(NUM_SAMPLES_PER_LETTER):
            try:
                f = random.choice(fonts)
                c = char.upper() if random.random() < 0.3 else char
                img_name = f"img_{i}.png"
                save_path = os.path.join(char_dir, img_name)
                img_vector = generate_char_image(c, f, save_path=save_path)
                all_images.append(img_vector)
                all_labels.append(label)
            except Exception as e:
                continue

    print(f"Генерация завершена. Всего {len(all_images)} изображений.")
    return np.array(all_images), np.array(all_labels)

def prepare_data(X, y, num_classes, test_size=0.2):
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_count = int(test_size * n_samples)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    def to_categorical(labels, num_classes):
        result = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            result[i, label] = 1.0
        return result

    y_train_cat = to_categorical(y_train, num_classes)

    X_train_arr = np.array([x.flatten() for x in X_train])
    X_test_arr = np.array([x.flatten() for x in X_test])
    y_train_idx = np.array([np.argmax(y) for y in y_train_cat]).astype('int64')
    y_test_idx = y_test.astype('int64')

    return X_train_arr, X_test_arr, y_train_idx, y_test_idx

# ============================================================================
# KERAS ПОДБОР ALPHA
# ============================================================================
def keras_alpha_search(X_train, X_test, y_train, y_test, num_classes):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("\n" + "="*60)
    print("ПОДБОР ЛУЧШЕГО ALPHA ДЛЯ KERAS")
    print("="*60)

    results = []

    for alpha_test in ALPHAS_TO_TEST:
        print(f"\n--- Тестирование alpha={alpha_test} ---")

        def make_custom_tanh(alpha):
            def custom_tanh(x):
                return tf.nn.tanh(alpha * x)
            return custom_tanh

        custom_tanh_func = make_custom_tanh(alpha_test)
        keras.utils.get_custom_objects()['custom_tanh_search'] = custom_tanh_func

        model = keras.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            layers.Flatten(),
            layers.Dense(128),
            layers.Activation(custom_tanh_func),
            layers.Dense(64),
            layers.Activation(custom_tanh_func),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            shuffle=True,
            callbacks=[early_stop],
            verbose=0
        )

        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        results.append({
            'alpha': alpha_test,
            'val_acc': best_val_acc,
            'val_loss': best_val_loss
        })
        print(f"alpha={alpha_test}: val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")

    best_result = max(results, key=lambda x: x['val_acc'])

    print("\n" + "="*60)
    print(f"ЛУЧШИЙ ALPHA ДЛЯ KERAS: {best_result['alpha']}")
    print(f"Точность: {best_result['val_acc']:.4f}")
    print(f"Потери: {best_result['val_loss']:.4f}")
    print("="*60)

    print("\nСводная таблица результатов Keras:")
    print("-" * 45)
    print(f"{'Alpha':<10} {'Val Acc':<15} {'Val Loss':<15}")
    print("-" * 45)
    for res in results:
        print(f"{res['alpha']:<10} {res['val_acc']:<15.4f} {res['val_loss']:<15.4f}")
    print("-" * 45)

    return best_result, results

# ============================================================================
# PYTORCH ПОДБОР ALPHA
# ============================================================================
def pytorch_alpha_search(X_train, X_test, y_train, y_test, num_classes, input_size):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")

    print("\n" + "="*60)
    print("ПОДБОР ЛУЧШЕГО ALPHA ДЛЯ PYTORCH")
    print("="*60)

    results = []

    for alpha_test in ALPHAS_TO_TEST:
        print(f"\n--- Тестирование alpha={alpha_test} ---")

        class PyTorchNetAlpha(nn.Module):
            def __init__(self, input_size, num_classes, alpha):
                super().__init__()
                self.alpha = alpha
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.xavier_uniform_(self.fc3.weight)

            def forward(self, x):
                x = torch.tanh(self.alpha * self.fc1(x))
                x = torch.tanh(self.alpha * self.fc2(x))
                x = self.fc3(x)
                return x

        X_train_pt = torch.FloatTensor(X_train).to(device)
        y_train_pt = torch.LongTensor(y_train).to(device)
        X_test_pt = torch.FloatTensor(X_test).to(device)
        y_test_pt = torch.LongTensor(y_test).to(device)

        train_dataset = TensorDataset(X_train_pt, y_train_pt)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = PyTorchNetAlpha(input_size, num_classes, alpha_test).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        best_val_acc = 0
        best_model_state = None
        patience = 15
        wait = 0

        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_pt)
                val_loss = criterion(val_outputs, y_test_pt).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_acc = (val_predicted == y_test_pt).sum().item() / len(y_test_pt)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        results.append({
            'alpha': alpha_test,
            'val_acc': best_val_acc,
            'val_loss': best_val_loss
        })
        print(f"alpha={alpha_test}: val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f}")

    best_result = max(results, key=lambda x: x['val_acc'])

    print("\n" + "="*60)
    print(f"ЛУЧШИЙ ALPHA ДЛЯ PYTORCH: {best_result['alpha']}")
    print(f"Точность: {best_result['val_acc']:.4f}")
    print(f"Потери: {best_result['val_loss']:.4f}")
    print("="*60)

    print("\nСводная таблица результатов PyTorch:")
    print("-" * 45)
    print(f"{'Alpha':<10} {'Val Acc':<15} {'Val Loss':<15}")
    print("-" * 45)
    for res in results:
        print(f"{res['alpha']:<10} {res['val_acc']:<15.4f} {res['val_loss']:<15.4f}")
    print("-" * 45)

    return best_result, results

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("ПОДБОР ГИПЕРПАРАМЕТРА ALPHA (tanh(alpha*x))")
    print("="*60)

    # Генерация данных
    random.seed(42)
    np.random.seed(42)

    X, y = generate_dataset()
    num_classes = len(ALPHABET)
    input_size = IMG_SIZE * IMG_SIZE

    X_train, X_test, y_train, y_test = prepare_data(X, y, num_classes)

    print(f"\nОбучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")

    # Подбор для Keras
    keras_best, keras_all = keras_alpha_search(
        X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1),
        X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1),
        y_train,
        y_test,
        num_classes
    )

    # Подбор для PyTorch
    pytorch_best, pytorch_all = pytorch_alpha_search(
        X_train, X_test, y_train, y_test, num_classes, input_size
    )

    # Итоговое сравнение
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*60)
    print(f"\nKeras - лучший alpha: {keras_best['alpha']}, точность: {keras_best['val_acc']:.4f}")
    print(f"PyTorch - лучший alpha: {pytorch_best['alpha']}, точность: {pytorch_best['val_acc']:.4f}")
    print("="*60)
