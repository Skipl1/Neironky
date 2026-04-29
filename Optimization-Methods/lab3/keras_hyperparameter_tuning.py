"""
Полноценный подбор гиперпараметров для Keras модели
Используем Keras Tuner для автоматического поиска
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================================
# КОНСТАНТЫ
# ============================================================================
IMG_SIZE = 32
ALPHABET = "йклмнопрсту"
NUM_SAMPLES_PER_LETTER = 1000
DATASET_DIR = "dataset_images"

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

# ============================================================================
# ПОДБОР ГИПЕРПАРАМЕТРОВ С KERAS TUNER
# ============================================================================
def create_model(hp):
    """Создание модели с гиперпараметрами для поиска"""
    
    # Гиперпараметр alpha для tanh(alpha*x)
    alpha = hp.Choice('alpha', values=[1, 2, 3, 4, 5])
    
    # Гиперпараметр learning_rate
    learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.0005, 0.001, 0.005])
    
    # Гиперпараметр количества нейронов в первом слое
    units_1 = hp.Choice('units_1', values=[64, 128, 256])
    
    # Гиперпараметр количества нейронов во втором слое
    units_2 = hp.Choice('units_2', values=[32, 64, 128])
    
    # Гиперпараметр dropout
    use_dropout = hp.Boolean('use_dropout')
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1) if use_dropout else 0
    
    # Гиперпараметр L2 регуляризации
    use_l2 = hp.Boolean('use_l2')
    l2_rate = hp.Float('l2_rate', min_value=0.0001, max_value=0.01, step=0.0001) if use_l2 else 0
    
    # Функция активации с параметром alpha
    def custom_tanh(x):
        return tf.nn.tanh(alpha * x)
    
    keras.utils.get_custom_objects()['custom_tanh_tuner'] = custom_tanh
    
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Flatten(),
        
        layers.Dense(units_1, kernel_regularizer=keras.regularizers.l2(l2_rate) if l2_rate > 0 else None),
        layers.Activation(custom_tanh),
        layers.Dropout(dropout_rate) if dropout_rate > 0 else layers.Activation('linear'),
        
        layers.Dense(units_2, kernel_regularizer=keras.regularizers.l2(l2_rate) if l2_rate > 0 else None),
        layers.Activation(custom_tanh),
        layers.Dropout(dropout_rate) if dropout_rate > 0 else layers.Activation('linear'),
        
        layers.Dense(len(ALPHABET), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Сохраняем гиперпараметры в модели
    model.hp_alpha = alpha
    model.hp_lr = learning_rate
    model.hp_units_1 = units_1
    model.hp_units_2 = units_2
    
    return model

def run_keras_tuner_search(X_train, y_train, X_val, y_val, max_trials=20):
    """Запуск поиска гиперпараметров"""
    
    try:
        import keras_tuner as kt
    except ImportError:
        print("Установите keras-tuner: pip install keras-tuner")
        return None
    
    print("\n" + "="*70)
    print("ПОДБОР ГИПЕРПАРАМЕТРОВ KERAS С KERAS TUNER")
    print("="*70)
    
    tuner = kt.Hyperband(
        hypermodel=create_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='keras_tuner_dir',
        project_name='char_recognition',
        overwrite=True
    )
    
    stop_early = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[stop_early],
        verbose=1
    )
    
    # Получаем лучшие гиперпараметры
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n" + "="*70)
    print("ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ:")
    print("="*70)
    print(f"  alpha:          {best_hps.get('alpha')}")
    print(f"  learning_rate:  {best_hps.get('learning_rate')}")
    print(f"  units_1:        {best_hps.get('units_1')}")
    print(f"  units_2:        {best_hps.get('units_2')}")
    print(f"  use_dropout:    {best_hps.get('use_dropout')}")
    if best_hps.get('use_dropout'):
        print(f"  dropout_rate:   {best_hps.get('dropout_rate')}")
    print(f"  use_l2:         {best_hps.get('use_l2')}")
    if best_hps.get('use_l2'):
        print(f"  l2_rate:        {best_hps.get('l2_rate')}")
    print("="*70)
    
    return tuner, best_hps

# ============================================================================
# РУЧНОЙ GRID SEARCH ПО ГИПЕРПАРАМЕТРАМ
# ============================================================================
def manual_grid_search(X_train, y_train, X_val, y_val):
    """Ручной поиск по сетке гиперпараметров"""
    
    print("\n" + "="*70)
    print("РУЧНОЙ GRID SEARCH ПО ГИПЕРПАРАМЕТРАМ")
    print("="*70)
    
    # Параметры для поиска
    alphas = [1, 2, 3]
    learning_rates = [0.0005, 0.001]
    units_options = [(128, 64), (256, 128), (128, 128)]
    
    results = []
    best_val_acc = 0
    best_params = {}
    
    total = len(alphas) * len(learning_rates) * len(units_options)
    count = 0
    
    for alpha in alphas:
        for lr in learning_rates:
            for units_1, units_2 in units_options:
                count += 1
                print(f"\n[{count}/{total}] alpha={alpha}, lr={lr}, units=({units_1}, {units_2})")
                
                def custom_tanh(x):
                    return tf.nn.tanh(alpha * x)
                
                keras.utils.get_custom_objects()['custom_tanh_grid'] = custom_tanh
                
                model = keras.Sequential([
                    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
                    layers.Flatten(),
                    layers.Dense(units_1),
                    layers.Activation(custom_tanh),
                    layers.Dense(units_2),
                    layers.Activation(custom_tanh),
                    layers.Dense(len(ALPHABET), activation='softmax')
                ])
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=0
                )
                
                val_acc = max(history.history['val_accuracy'])
                val_loss = min(history.history['val_loss'])
                
                results.append({
                    'alpha': alpha,
                    'learning_rate': lr,
                    'units_1': units_1,
                    'units_2': units_2,
                    'val_acc': val_acc,
                    'val_loss': val_loss
                })
                
                print(f"  -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {
                        'alpha': alpha,
                        'learning_rate': lr,
                        'units_1': units_1,
                        'units_2': units_2,
                        'val_acc': val_acc,
                        'val_loss': val_loss
                    }
    
    print("\n" + "="*70)
    print("ЛУЧШИЕ ПАРАМЕТРЫ (Grid Search):")
    print("="*70)
    print(f"  alpha:         {best_params['alpha']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  units_1:       {best_params['units_1']}")
    print(f"  units_2:       {best_params['units_2']}")
    print(f"  val_accuracy:  {best_params['val_acc']:.4f}")
    print(f"  val_loss:      {best_params['val_loss']:.4f}")
    print("="*70)
    
    # Таблица результатов
    print("\nТаблица всех результатов:")
    print("-" * 80)
    print(f"{'Alpha':<6} {'LR':<10} {'Units':<15} {'Val Acc':<12} {'Val Loss':<12}")
    print("-" * 80)
    for res in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        print(f"{res['alpha']:<6} {res['learning_rate']:<10} ({res['units_1']},{res['units_2']:<8}) {res['val_acc']:<12.4f} {res['val_loss']:<12.4f}")
    print("-" * 80)
    
    return best_params, results

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("ПОДБОР ГИПЕРПАРАМЕТРОВ ДЛЯ KERAS МОДЕЛИ")
    print("="*70)
    
    # Фиксируем random seed для воспроизводимости
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Генерация данных
    X, y = generate_dataset()
    num_classes = len(ALPHABET)
    
    # Разделение на train/val/test
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = 0.2
    val_size = 0.1
    
    test_count = int(test_size * n_samples)
    val_count = int(val_size * n_samples)
    
    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    train_indices = indices[test_count + val_count:]
    
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    
    # Подготовка данных для Keras
    X_train_k = np.array([x.reshape(IMG_SIZE, IMG_SIZE, 1) for x in X_train]).astype('float32')
    X_val_k = np.array([x.reshape(IMG_SIZE, IMG_SIZE, 1) for x in X_val]).astype('float32')
    X_test_k = np.array([x.reshape(IMG_SIZE, IMG_SIZE, 1) for x in X_test]).astype('float32')
    
    y_train_idx = y_train.astype('int64')
    y_val_idx = y_val.astype('int64')
    y_test_idx = y_test.astype('int64')
    
    print(f"\nОбучающая: {len(X_train_k)}, Валидация: {len(X_val_k)}, Тест: {len(X_test_k)}")
    
    # Запуск ручного grid search
    best_params, all_results = manual_grid_search(
        X_train_k, y_train_idx,
        X_val_k, y_val_idx
    )
    
    # Финальное обучение с лучшими параметрами
    print("\n" + "="*70)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ С ЛУЧШИМИ ПАРАМЕТРАМИ")
    print("="*70)
    
    def final_custom_tanh(x):
        return tf.nn.tanh(best_params['alpha'] * x)
    
    keras.utils.get_custom_objects()['custom_tanh_final'] = final_custom_tanh
    
    final_model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Flatten(),
        layers.Dense(best_params['units_1']),
        layers.Activation(final_custom_tanh),
        layers.Dense(best_params['units_2']),
        layers.Activation(final_custom_tanh),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nАрхитектура с лучшими параметрами:")
    final_model.summary()
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    final_history = final_model.fit(
        X_train_k, y_train_idx,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_k, y_val_idx),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Оценка на тесте
    test_loss, test_acc = final_model.evaluate(X_test_k, y_test_idx, verbose=0)
    
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"Лучшие гиперпараметры:")
    print(f"  alpha:         {best_params['alpha']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  units_1:       {best_params['units_1']}")
    print(f"  units_2:       {best_params['units_2']}")
    print(f"\nТочность на валидации: {best_params['val_acc']:.4f}")
    print(f"Точность на тесте:      {test_acc:.4f}")
    print("="*70)
    
    # Сохранение модели
    final_model.save('keras_best_model.h5')
    print("\nМодель сохранена в keras_best_model.h5")
