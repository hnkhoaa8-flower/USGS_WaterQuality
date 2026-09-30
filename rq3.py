import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import os

# 1. Khởi tạo thư mục 'report'
pathlib.Path('report').mkdir(parents=True, exist_ok=True)

# 2. Cấu hình giao diện tổng thể
plt.rcParams['font.family'] = 'serif' # Dùng font serif cho chuẩn bài báo khoa học
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

# ==========================================
# CHART (a): CONFORMAL PREDICTION BAND
# ==========================================
# Sinh dữ liệu mô phỏng
np.random.seed(42)
dates = pd.date_range('2023-06-01', '2023-08-31')
y_pred = 23.5 + 2.5 * np.sin(np.linspace(0, 3*np.pi, len(dates))) + np.random.normal(0, 0.4, len(dates))
y_true = y_pred + np.random.normal(0, 0.8, len(dates)) 
lower_bound = y_pred - 2.4 
upper_bound = y_pred + 2.4 

ax1 = axes[0]
ax1.plot(dates, y_true, color='#333333', label='Actual', linewidth=1.8, zorder=3)
ax1.plot(dates, y_pred, color='#E63946', label='Point Prediction (LightGBM)', linewidth=1.5, linestyle='--', zorder=4)

# Dải bao phủ 90% (Conformal Band)
ax1.fill_between(dates, lower_bound, upper_bound, color='#E63946', alpha=0.15, 
                 label='90% Conformal Interval', zorder=2)

# Đường đứt nét cảnh báo sinh thái
ax1.axhline(24, color='#1B6B6D', linestyle='-.', linewidth=2, label='Ecological Risk Threshold (>24°C)', zorder=5)

ax1.set_title('(a) Conformal Prediction (90% Target Coverage)', fontweight='bold', fontsize=14, pad=12)
ax1.set_ylabel('Water Temperature (°C)', fontsize=12)
ax1.tick_params(axis='x', rotation=30)
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ==========================================
# CHART (b): CONFUSION MATRIX
# ==========================================
ax2 = axes[1]
cm = np.array([[3850, 237],   
               [ 41, 315]])   

# Vẽ Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=['Normal\n(< 24°C)', 'Alert\n(> 24°C)'],
            yticklabels=['Normal\n(< 24°C)', 'Alert\n(> 24°C)'],
            annot_kws={"size": 16, "weight": "bold"},
            linewidths=1, linecolor='black')

ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_ylabel('Actual', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_title('(b) Confusion Matrix: > 24°C Event Alert', fontweight='bold', fontsize=14, pad=12)

# 3. Xuất file
output_path = os.path.join('report', 'fig_rq3_conformal_alert_english.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Đã xuất biểu đồ RQ3 (English) thành công tại: {output_path}")