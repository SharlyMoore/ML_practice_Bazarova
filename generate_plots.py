import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings('ignore')

# Настройка базового стиля
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 150

data = {
    'Модель': ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'],
    'Обнаружено объектов (среднее)': [16.4, 14.2, 16.1, 15.3, 16.1],
    'Средняя уверенность': [0.60, 0.64, 0.68, 0.75, 0.79],
    'Время, с (среднее)': [3.78, 3.56, 4.38, 8.81, 13.42],
    'Параметры (M)': [3.2, 11.2, 25.9, 43.7, 68.2],
    'FLOPs (B)': [8.7, 28.6, 78.9, 165.2, 257.8],
    'Размер весов (MB)': [6.2, 22.5, 52.0, 88.0, 137.0]
}

df = pd.DataFrame(data)
models = df['Модель'].tolist()

if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
    print("Создана папка: results/plots")

print("=" * 60)
print("Начинаем генерацию графиков...")
print("=" * 60)


# ГРАФИК 1: Средняя уверенность моделей
def plot_confidence():
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = plt.bar(models, df['Средняя уверенность'], color=colors, 
                   edgecolor='black', linewidth=1.2, width=0.7)
    
    for bar, val in zip(bars, df['Средняя уверенность']):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.015, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Модель', fontsize=14, fontweight='bold')
    plt.ylabel('Средняя уверенность', fontsize=14, fontweight='bold')
    plt.title('Рис. 1. Сравнение средней уверенности моделей YOLOv8', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Порог уверенности (0.5)')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/plots/fig1_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График 1 сохранен: results/plots/fig1_confidence.png")

# ГРАФИК 2: Время обработки изображения
def plot_inference_time():
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = plt.bar(models, df['Время, с (среднее)'], color=colors,
                   edgecolor='black', linewidth=1.2, width=0.7)
    
    for bar, val in zip(bars, df['Время, с (среднее)']):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                f'{val:.2f} с', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Модель', fontsize=14, fontweight='bold')
    plt.ylabel('Время инференса, секунды', fontsize=14, fontweight='bold')
    plt.title('Рис. 2. Сравнение времени инференса моделей YOLOv8', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/fig2_inference_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График 2 сохранен: results/plots/fig2_inference_time.png")

# ГРАФИК 3: Количество обнаруженных объектов
def plot_detected_objects():
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = plt.bar(models, df['Обнаружено объектов (среднее)'], color=colors,
                   edgecolor='black', linewidth=1.2, width=0.7)
    
    for bar, val in zip(bars, df['Обнаружено объектов (среднее)']):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.3, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Модель', fontsize=14, fontweight='bold')
    plt.ylabel('Среднее количество объектов', fontsize=14, fontweight='bold')
    plt.title('Рис. 3. Количество обнаруженных объектов на изображение', fontsize=16, fontweight='bold')
    plt.ylim(0, 20)
    plt.grid(axis='y', alpha=0.3)
    
    mean_val = np.mean(df['Обнаружено объектов (среднее)'])
    plt.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
               label=f'Среднее: {mean_val:.1f}')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/plots/fig3_detected_objects.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График 3 сохранен: results/plots/fig3_detected_objects.png")


# ГРАФИК 4: Комплексный анализ
def plot_comprehensive():
    plt.figure(figsize=(12, 7))
    
    # Нормализуем данные
    confidence_norm = df['Средняя уверенность'] / df['Средняя уверенность'].max()
    time_norm = 1 - (df['Время, с (среднее)'] / df['Время, с (среднее)'].max())
    objects_norm = df['Обнаружено объектов (среднее)'] / df['Обнаружено объектов (среднее)'].max()
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, confidence_norm, width, label='Уверенность (норм.)', 
            color='#3498db', edgecolor='black', linewidth=1)
    plt.bar(x, time_norm, width, label='Скорость (норм., выше=быстрее)', 
            color='#2ecc71', edgecolor='black', linewidth=1)
    plt.bar(x + width, objects_norm, width, label='Количество объектов (норм.)', 
            color='#e74c3c', edgecolor='black', linewidth=1)
    
    plt.xlabel('Модель', fontsize=14, fontweight='bold')
    plt.ylabel('Нормированное значение', fontsize=14, fontweight='bold')
    plt.title('Рис. 4. Комплексное сравнение моделей YOLOv8', fontsize=16, fontweight='bold')
    plt.xticks(x, models)
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('results/plots/fig4_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График 4 сохранен: results/plots/fig4_comprehensive.png")

# ГРАФИК 5: Соотношение точность vs скорость
def plot_accuracy_speed_tradeoff():
    plt.figure(figsize=(10, 8))
    
    sizes = np.array(df['Параметры (M)']) * 8
    
    scatter = plt.scatter(df['Время, с (среднее)'], df['Средняя уверенность'], 
                         s=sizes, c=range(len(models)), cmap='viridis', 
                         alpha=0.8, edgecolors='black', linewidth=1.5)
    
    for i, model in enumerate(models):
        plt.annotate(model, 
                   (df['Время, с (среднее)'][i], df['Средняя уверенность'][i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlabel('Время инференса, с', fontsize=14, fontweight='bold')
    plt.ylabel('Средняя уверенность', fontsize=14, fontweight='bold')
    plt.title('Рис. 5. Соотношение точности и скорости для моделей YOLOv8', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Размер модели (0=n, 4=x)', fontsize=12)
    
    # Тренд
    z = np.polyfit(df['Время, с (среднее)'], df['Средняя уверенность'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(df['Время, с (среднее)']) - 0.5, 
                          max(df['Время, с (среднее)']) + 0.5, 100)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2, label='Тренд')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results/plots/fig5_accuracy_speed_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График 5 сохранен: results/plots/fig5_accuracy_speed_tradeoff.png")

# ГРАФИК 6: Индекс эффективности
def plot_efficiency_index():
    plt.figure(figsize=(10, 6))
    
    efficiency = df['Средняя уверенность'] / df['Время, с (среднее)']
    
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = plt.bar(models, efficiency, color=colors, 
                   edgecolor='black', linewidth=1.2, width=0.7)
    
    for bar, val in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.002, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Модель', fontsize=14, fontweight='bold')
    plt.ylabel('Индекс эффективности', fontsize=14, fontweight='bold')
    plt.title('Рис. 6. Индекс эффективности моделей YOLOv8\n(выше = лучше баланс качества и скорости)', 
             fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    max_idx = np.argmax(efficiency)
    bars[max_idx].set_edgecolor('gold')
    bars[max_idx].set_linewidth(3)
    plt.text(bars[max_idx].get_x() + bars[max_idx].get_width()/2, 
            bars[max_idx].get_height() + 0.005, 
            '★ Лучшая', ha='center', va='bottom', fontweight='bold', color='gold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/plots/fig6_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ График 6 сохранен: results/plots/fig6_efficiency.png")

# ГРАФИК 7: Сводная таблица
def create_summary_table():
    plt.figure(figsize=(12, 4))
    plt.axis('tight')
    plt.axis('off')
    
    headers = ['Модель', 'Объектов', 'Уверенность', 'Время (с)', 'Параметры (M)']
    table_data = []
    
    for i, row in df.iterrows():
        table_data.append([
            row['Модель'],
            f"{row['Обнаружено объектов (среднее)']:.1f}",
            f"{row['Средняя уверенность']:.2f}",
            f"{row['Время, с (среднее)']:.2f}",
            f"{row['Параметры (M)']:.1f}"
        ])
    
    table = plt.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Таблица 1. Сводные результаты экспериментов', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/plots/table_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Таблица сохранена: results/plots/table_summary.png")

if __name__ == "__main__":
    try:
        plot_confidence()
        plot_inference_time()
        plot_detected_objects()
        plot_comprehensive()
        plot_accuracy_speed_tradeoff()
        plot_efficiency_index()
        create_summary_table()
        
        print("\n" + "=" * 60)
        print("ВСЕ ГРАФИКИ УСПЕШНО СОЗДАНЫ!")
        print(f"Папка: {os.path.abspath('results/plots')}")
        
        # Список созданных файлов
        files = os.listdir('results/plots')
        print(f"\nСоздано файлов: {len(files)}")
        for f in sorted(files):
            if f.endswith('.png'):
                print(f"  - {f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("Проверьте, что у вас установлены библиотеки:")
        print("pip install matplotlib numpy pandas")