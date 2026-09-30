import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(10, 6))

models = [
    'Persistence (Naive)', 
    'Ridge Regression', 
    'Long-horizon Linear', 
    'LightGBM (Default)', 
    'LightGBM (Tuned)', 
    'TimesFM (Zero-shot)', 
    'Chronos (Zero-shot)', 
    'Sundial (Zero-shot)'
]
maes = [1.563, 1.496, 1.502, 1.485, 1.453, 1.553, 1.618, 1.745]

# Tô màu nhấn mạnh LightGBM tinh chỉnh, các mô hình khác màu xám/xanh trầm
colors = ['#1B6B6D', '#1B6B6D', '#1B6B6D', '#1B6B6D', '#F0A268', '#1B6B6D', '#1B6B6D', '#1B6B6D']

bars = ax.barh(models, maes, color=colors, edgecolor='black', linewidth=0.7)
ax.set_xlim(1.3, 1.8) # Cắt trục X để thấy rõ khoảng chênh lệch
ax.set_xlabel('Mean Absolute Error (MAE - °C)', fontsize=11, fontweight='bold')
ax.set_title('Comparison of Forecasting Accuracy (MAE) Across Predictive Models', fontsize=13, fontweight='bold', pad=15)

# Gắn giá trị số trực tiếp lên đầu mỗi cột
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('report/fig_rq2_model_comparison_bar.png', dpi=300, bbox_inches='tight')
print("Successfully exported RQ2 bar chart!")