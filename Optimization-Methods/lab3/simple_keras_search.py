"""
Простой подбор гиперпараметров для Keras
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Параметры
IMG_SIZE = 32
num_classes = 11
ALPHABET = "йклмнопрсту"

# Загрузка данных (предполагается, что они уже есть)
# X_train_k, y_train_idx, X_val_k, y_val_idx - должны быть подготовлены

print("="*50)
print("ПОДБОР ГИПЕРПАРАМЕТРОВ KERAS")
print("="*50)

best_acc = 0
best_params = {}

# Перебор параметров
for alpha in [1, 2, 3]:
    for lr in [0.0005, 0.001]:
        for units in [(128, 64), (256, 128)]:
            print(f"\nalpha={alpha}, lr={lr}, units={units}")
            
            def custom_tanh(x):
                return tf.nn.tanh(alpha * x)
            
            keras.utils.get_custom_objects()['custom_tanh'] = custom_tanh
            
            model = keras.Sequential([
                layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
                layers.Flatten(),
                layers.Dense(units[0]),
                layers.Activation(custom_tanh),
                layers.Dense(units[1]),
                layers.Activation(custom_tanh),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train_k, y_train_idx,
                epochs=30,
                batch_size=32,
                validation_data=(X_val_k, y_val_idx),
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            print(f"  -> val_acc = {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'alpha': alpha, 'lr': lr, 'units': units, 'val_acc': val_acc}

print("\n" + "="*50)
print("ЛУЧШИЕ ПАРАМЕТРЫ:")
print(f"  alpha  = {best_params['alpha']}")
print(f"  lr     = {best_params['lr']}")
print(f"  units  = {best_params['units']}")
print(f"  acc    = {best_params['val_acc']:.4f}")
print("="*50)
