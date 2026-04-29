# -*- coding: utf-8 -*-
"""
Скрипт для генерации всех изображений для отчёта по лабораторной работе №2
Neural Networks for Car Evaluation Dataset
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import os
import warnings
warnings.filterwarnings('ignore')

# Установка случайного зерна для воспроизводимости
my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)

# Пути
IMAGES_DIR = r"d:\Для Windows 11\Всё подряд\Users\kreck\Desktop\Прочее\Дз по нейросетям\Отчёты\Optimization-Methods\lab2\images"
CARS_PATH = r"d:\Для Windows 11\Всё подряд\Users\kreck\Desktop\Прочее\Дз по нейросетям\ioy\cars.csv"

os.makedirs(IMAGES_DIR, exist_ok=True)

def save_fig(name, **kwargs):
    path = os.path.join(IMAGES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', **kwargs)
    plt.close()

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# 1. Загрузка и подготовка данных
# ============================================================================

print("Загрузка данных...")

cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(CARS_PATH, header=0, names=cols)
df = df[df['class'] != 'class'].reset_index(drop=True)

# Преобразование категорий в числа
mappings_per_col = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint':  {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors':  {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

df_encoded = df.copy()
for col in df_encoded.columns[:-1]:
    df_encoded[col] = df_encoded[col].map(mappings_per_col[col]).fillna(0).astype(int)

df_encoded['class'] = df_encoded['class'].map(class_mapping)

# Визуализация данных (3D scatter plot)
print("Создание визуализации данных...")
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

np.random.seed(42)
x = df_encoded['buying'].values + np.random.uniform(-0.1, 0.1, len(df_encoded))
y = df_encoded['safety'].values + np.random.uniform(-0.1, 0.1, len(df_encoded))
z = df_encoded['persons'].values + np.random.uniform(-0.1, 0.1, len(df_encoded))

colors = df_encoded['class'].map({0: 'red', 1: 'blue', 2: 'green', 3: 'purple'})

ax.scatter(x, y, z, c=colors, alpha=0.6, s=30)
ax.set_xlabel('Buying (Цена)', fontsize=12)
ax.set_ylabel('Safety (Безопасность)', fontsize=12)
ax.set_zlabel('Persons (Вместимость)', fontsize=12)
ax.set_title('Визуализация данных (Buying vs Safety vs Persons)', fontsize=14)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='unacc'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='acc'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='good'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='vgood')
]
ax.legend(handles=legend_elements, loc='best')

save_fig('data_visualization.png')

# ============================================================================
# 2. Разделение данных и масштабирование
# ============================================================================

X_raw = df_encoded.iloc[:, :-1].values
y_raw = df_encoded.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_raw, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестирующей выборки: {len(X_test)}")

# ============================================================================
# 3. Персептрон
# ============================================================================

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=5000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

print("\n" + "="*60)
print("Персептрон")
print("="*60)

perceptron_results = {}
unique_classes = np.unique(y_raw)

for target_class in unique_classes:
    print(f"\nКласс: {target_class} ({reverse_class_mapping[target_class]})")
    
    y_train_binary = np.where(y_train == target_class, 1, 0)
    y_test_binary = np.where(y_test == target_class, 1, 0)
    
    model = Perceptron(learning_rate=0.01, n_iters=5000)
    model.fit(X_train, y_train_binary)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train_binary, y_train_pred)
    test_acc = accuracy_score(y_test_binary, y_test_pred)
    test_rec = recall_score(y_test_binary, y_test_pred, zero_division=0)
    test_prec = precision_score(y_test_binary, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test_binary, y_test_pred, zero_division=0)
    
    perceptron_results[target_class] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_rec': test_rec,
        'test_prec': test_prec,
        'test_f1': test_f1,
        'model': model
    }
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    
    # Визуализация
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red' if c == target_class else 'blue' for c in y_test]
    ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=colors, alpha=0.6, s=30)
    
    def f(x, y, model):
        if model.weights[2] == 0:
            return np.zeros_like(x)
        return (-model.bias - x*model.weights[0] - y*model.weights[1]) / model.weights[2]
    
    x_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 30)
    y_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 30)
    x11, x22 = np.meshgrid(x_range, y_range)
    z_plane = f(x11, x22, model)
    
    ax.plot_surface(x11, x22, z_plane, alpha=0.3, color='gray')
    
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.set_title(f'Разделяющая плоскость для класса: {reverse_class_mapping[target_class]}')
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red' if target_class == 0 else 'blue', markersize=10, label=f'Class {reverse_class_mapping[target_class]}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue' if target_class == 0 else 'red', markersize=10, label='Other classes')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    save_fig(f'perceptron_class_{reverse_class_mapping[target_class]}.png')

# Матрицы ошибок для персептрона
print("\nСоздание матриц ошибок для персептрона...")

perceptron_train_pred = np.zeros(len(X_train))
perceptron_test_pred = np.zeros(len(X_test))

for i in range(len(X_train)):
    scores = [np.dot(X_train[i], perceptron_results[c]['model'].weights) + perceptron_results[c]['model'].bias for c in unique_classes]
    perceptron_train_pred[i] = unique_classes[np.argmax(scores)]

for i in range(len(X_test)):
    scores = [np.dot(X_test[i], perceptron_results[c]['model'].weights) + perceptron_results[c]['model'].bias for c in unique_classes]
    perceptron_test_pred[i] = unique_classes[np.argmax(scores)]

cm_train = confusion_matrix(y_train, perceptron_train_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='magma', 
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Perceptron (Train)')
save_fig('perceptron_confusion_train.png')

cm_test = confusion_matrix(y_test, perceptron_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Perceptron (Test)')
save_fig('perceptron_confusion_test.png')

# ============================================================================
# 4. Сеть Кохонена
# ============================================================================

print("\n" + "="*60)
print("Сеть Кохонена (SOM)")
print("="*60)

class SimpleKohonen:
    def __init__(self, n_clusters=4, n_iterations=100, learning_rate=0.5):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.lr = learning_rate
        self.weights = None
    
    def fit(self, X):
        n_samples, n_features = X.shape
        np.random.seed(42)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.weights = X[indices].copy()
        
        for _ in range(self.n_iterations):
            for sample in X:
                distances = np.linalg.norm(self.weights - sample, axis=1)
                winner = np.argmin(distances)
                self.weights[winner] += self.lr * (sample - self.weights[winner])
            self.lr *= 0.99
        return self
    
    def predict(self, X):
        distances = np.linalg.norm(self.weights[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=0)

kohonen = SimpleKohonen(n_clusters=4, n_iterations=100, learning_rate=0.5)
kohonen.fit(X_train)

kohonen_train_pred = kohonen.predict(X_train)
kohonen_test_pred = kohonen.predict(X_test)

cluster_to_class = {}
for cluster in range(4):
    mask = kohonen_train_pred == cluster
    if np.sum(mask) > 0:
        classes, counts = np.unique(y_train[mask], return_counts=True)
        cluster_to_class[cluster] = classes[np.argmax(counts)]

kohonen_train_mapped = np.array([cluster_to_class.get(c, 0) for c in kohonen_train_pred])
kohonen_test_mapped = np.array([cluster_to_class.get(c, 0) for c in kohonen_test_pred])

print(f"Kohonen Train Accuracy: {accuracy_score(y_train, kohonen_train_mapped):.4f}")
print(f"Kohonen Test Accuracy: {accuracy_score(y_test, kohonen_test_mapped):.4f}")

cm_kohonen_train = confusion_matrix(y_train, kohonen_train_mapped)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_kohonen_train, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Kohonen (Train)')
save_fig('kohonen_confusion_train.png')

cm_kohonen_test = confusion_matrix(y_test, kohonen_test_mapped)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_kohonen_test, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Kohonen (Test)')
save_fig('kohonen_confusion_test.png')

# ============================================================================
# 5. PNN
# ============================================================================

print("\n" + "="*60)
print("PNN (Probabilistic Neural Network)")
print("="*60)

class SimplePNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None
        self.classes = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self
    
    def predict(self, X):
        predictions = []
        for sample in X:
            probabilities = []
            for c in self.classes:
                mask = self.y_train == c
                X_c = self.X_train[mask]
                distances = np.linalg.norm(X_c - sample, axis=1)
                prob = np.mean(np.exp(-distances**2 / (2 * self.sigma**2)))
                probabilities.append(prob)
            predictions.append(self.classes[np.argmax(probabilities)])
        return np.array(predictions)

print("Подбор оптимального sigma для PNN...")
sigmas = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
best_sigma = 0.1
best_acc = 0

for sigma in sigmas:
    pnn = SimplePNN(sigma=sigma)
    pnn.fit(X_train, y_train)
    y_pred = pnn.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    if acc > best_acc:
        best_acc = acc
        best_sigma = sigma

print(f"Лучший sigma: {best_sigma}, Train Accuracy: {best_acc:.4f}")

pnn = SimplePNN(sigma=best_sigma)
pnn.fit(X_train, y_train)

pnn_train_pred = pnn.predict(X_train)
pnn_test_pred = pnn.predict(X_test)

print(f"PNN Train Accuracy: {accuracy_score(y_train, pnn_train_pred):.4f}")
print(f"PNN Test Accuracy: {accuracy_score(y_test, pnn_test_pred):.4f}")

cm_pnn_train = confusion_matrix(y_train, pnn_train_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_pnn_train, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: PNN (Train)')
save_fig('pnn_confusion_train.png')

cm_pnn_test = confusion_matrix(y_test, pnn_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_pnn_test, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: PNN (Test)')
save_fig('pnn_confusion_test.png')

# ============================================================================
# 6. MLP (scikit-learn)
# ============================================================================

print("\n" + "="*60)
print("MLP (scikit-learn)")
print("="*60)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

mlp = MLPClassifier(random_state=42, max_iter=500)
grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")

best_mlp = grid_search.best_estimator_

mlp_train_pred = best_mlp.predict(X_train)
mlp_test_pred = best_mlp.predict(X_test)

print(f"MLP Train Accuracy: {accuracy_score(y_train, mlp_train_pred):.4f}")
print(f"MLP Test Accuracy: {accuracy_score(y_test, mlp_test_pred):.4f}")

cm_mlp_train = confusion_matrix(y_train, mlp_train_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp_train, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: MLP sklearn (Train)')
save_fig('mlp_sklearn_confusion_train.png')

cm_mlp_test = confusion_matrix(y_test, mlp_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp_test, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: MLP sklearn (Test)')
save_fig('mlp_sklearn_confusion_test.png')

# ============================================================================
# 7. MLP (Keras-like)
# ============================================================================

print("\n" + "="*60)
print("MLP (Keras-like)")
print("="*60)

mlp_keras = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='constant',
    random_state=42,
    max_iter=100,
    early_stopping=False
)

mlp_keras.fit(X_train, y_train)

mlp_keras_train_pred = mlp_keras.predict(X_train)
mlp_keras_test_pred = mlp_keras.predict(X_test)

print(f"MLP Keras Train Accuracy: {accuracy_score(y_train, mlp_keras_train_pred):.4f}")
print(f"MLP Keras Test Accuracy: {accuracy_score(y_test, mlp_keras_test_pred):.4f}")

cm_mlp_keras_train = confusion_matrix(y_train, mlp_keras_train_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp_keras_train, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: MLP Keras (Train)')
save_fig('mlp_keras_confusion_train.png')

cm_mlp_keras_test = confusion_matrix(y_test, mlp_keras_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp_keras_test, annot=True, fmt='d', cmap='magma',
            xticklabels=['unacc', 'acc', 'good', 'vgood'],
            yticklabels=['unacc', 'acc', 'good', 'vgood'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: MLP Keras (Test)')
save_fig('mlp_keras_confusion_test.png')

# ============================================================================
# 8. Сравнение методов
# ============================================================================

print("\n" + "="*60)
print("Сравнение методов")
print("="*60)

methods = ['Perceptron', 'Kohonen', 'PNN', 'MLP sklearn', 'MLP Keras']
train_accs = [
    accuracy_score(y_train, perceptron_train_pred),
    accuracy_score(y_train, kohonen_train_mapped),
    accuracy_score(y_train, pnn_train_pred),
    accuracy_score(y_train, mlp_train_pred),
    accuracy_score(y_train, mlp_keras_train_pred)
]
test_accs = [
    accuracy_score(y_test, perceptron_test_pred),
    accuracy_score(y_test, kohonen_test_mapped),
    accuracy_score(y_test, pnn_test_pred),
    accuracy_score(y_test, mlp_test_pred),
    accuracy_score(y_test, mlp_keras_test_pred)
]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, train_accs, width, label='Train', color='steelblue')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='coral')

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Comparison of Classification Methods', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.legend()
ax.set_ylim(0, 1.1)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
save_fig('methods_comparison.png')

print("\nВсе изображения сохранены в папку 'images/'")
print("Готово!")
