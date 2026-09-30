"""
Hình bản đồ mục 3.1 (v2): 37 trạm USGS trên nền vệ tinh.
- Bản đồ chính bên trái; hai khung phóng to A, B và bảng số->mã trạm ở cột phải
  (khung zoom nằm NGOÀI bản đồ nên không thể che trạm nào)
- A = cụm Harrisburg-Lebanon-Lancaster, B = cụm Philadelphia (đổi trong ZOOMS)
- Trạm trong khung A/B chỉ đánh dấu chữ A/B trên bản đồ chính, số thứ tự nằm ở khung zoom
- Nhãn số được adjustText đẩy ra xa cả các điểm trạm khác để không đè nhau

Cài đặt: pip install contextily adjustText pyproj pandas matplotlib
Chạy từ thư mục gốc dự án: python fig_data_map_v2.py
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import contextily as ctx
from adjustText import adjust_text
from pyproj import Transformer

# ---------------- cấu hình ----------------
SITES_CSV = 'data/raw/wq_sites_all_valid.csv'
FEAT_PARQUET = 'data/processed/feat.parquet'
OUT = 'report/fig_data_map.png'
BASEMAP = ctx.providers.Esri.WorldImagery
# (lon_min, lat_min, lon_max, lat_max)
ZOOMS = {'A': ('Harrisburg - Lebanon - Lancaster', (-77.05, 39.85, -76.15, 40.50)),
         'B': ('Philadelphia area', (-75.45, 39.85, -75.05, 40.25))}
COLORS = {'PA': '#ff595e', 'MD': '#ffca3a', 'NY': '#8ac926', 'WV': '#1982c4', 
          'DC': '#f15bb5', 'DE': '#c77dff', 'VA': '#00f5d4'}

# ---------------- dữ liệu ----------------
pathlib.Path('report').mkdir(exist_ok=True)
s = pd.read_csv(SITES_CSV, dtype={'site_no': str})
s['site_no'] = s['site_no'].apply(lambda x: x.zfill(8) if len(x) == 7 else x)
if pathlib.Path(FEAT_PARQUET).exists():
    used = set(pd.read_parquet(FEAT_PARQUET, columns=['station'])['station'].astype(str).unique())
    s = s[s['site_no'].isin(used)]
s = (s.dropna(subset=['dec_lat_va', 'dec_long_va'])
       .sort_values(['state', 'dec_long_va']).reset_index(drop=True))
s['num'] = np.arange(1, len(s) + 1)

tf = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
s['x'], s['y'] = tf.transform(s['dec_long_va'].values, s['dec_lat_va'].values)

boxes, s['zoom'] = {}, ''
for k, (title, (lo0, la0, lo1, la1)) in ZOOMS.items():
    x0, y0 = tf.transform(lo0, la0); x1, y1 = tf.transform(lo1, la1)
    boxes[k] = (x0, y0, x1, y1)
    s.loc[s['x'].between(x0, x1) & s['y'].between(y0, y1), 'zoom'] = k

# ---------------- hàm dùng chung ----------------
def draw_points(ax, df, label_df, all_pts, size=70, fs=8, tag_col=None):
    """Vẽ điểm; nhãn là số (hoặc chữ tag_col); đẩy nhãn tránh mọi điểm trong all_pts."""
    for st, g in df.groupby('state'):
        ax.scatter(g['x'], g['y'], s=size, c=COLORS.get(st, 'white'),
                   edgecolor='white', linewidth=0.9, zorder=5)
    texts = [ax.text(r.x, r.y, str(r.num), fontsize=fs, fontweight='bold', color='black',
                     zorder=7, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.9))
             for r in label_df.itertuples()]
    if texts:
        adjust_text(texts, ax=ax, x=all_pts[:, 0], y=all_pts[:, 1],
                    expand=(1.8, 2.0), force_text=(0.7, 0.9), force_static=(0.8, 1.0),
                    arrowprops=dict(arrowstyle='-', color='white', lw=0.7, shrinkA=0, shrinkB=4),
                    ensure_inside_axes=True)

def set_view(ax, x0, x1, y0, y1):
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal', adjustable='box')

inv = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
def deg_ticks(ax, n=4):
    xt = np.linspace(*ax.get_xlim(), n); yt = np.linspace(*ax.get_ylim(), n)
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([f'{inv.transform(x, ax.get_ylim()[0])[0]:.1f}°' for x in xt], fontsize=8)
    ax.set_yticklabels([f'{inv.transform(ax.get_xlim()[0], y)[1]:.1f}°' for y in yt], fontsize=8)

# ---------------- khung hình ----------------
fig = plt.figure(figsize=(13.5, 9.5))
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.5], wspace=0.05)
ax = fig.add_subplot(gs[0])
gr = gs[1].subgridspec(len(ZOOMS) + 1, 1, height_ratios=[1] * len(ZOOMS) + [0.85], hspace=0.12)
all_pts = s[['x', 'y']].to_numpy()

# ---- bản đồ chính ----
padx = (s['x'].max() - s['x'].min()) * 0.07; pady = (s['y'].max() - s['y'].min()) * 0.10
set_view(ax, s['x'].min() - padx, s['x'].max() + padx, s['y'].min() - pady, s['y'].max() + pady)
ctx.add_basemap(ax, source=BASEMAP, crs='EPSG:3857', attribution_size=6)
draw_points(ax, s, s[s['zoom'] == ''], all_pts)
for k, (x0, y0, x1, y1) in boxes.items():
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='white', lw=1.0, zorder=6))
    ax.text(x0, y1, k, color='black', fontsize=10, fontweight='bold', ha='left', va='bottom', zorder=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))
deg_ticks(ax)
ax.set_title('USGS continuous stations used in the study (n = %d)' % len(s), fontsize=11)

h = [plt.Line2D([], [], marker='o', ls='', mfc=COLORS.get(st, 'white'), mec='k', ms=8,
                label=f'{st} ({n})') for st, n in s['state'].value_counts().items()]
ax.legend(handles=h, loc='upper left', fontsize=8, framealpha=0.9,
          title='State (stations)', title_fontsize=8)

# ---- các khung zoom (ngoài bản đồ) ----
for i, (k, (title, _)) in enumerate(ZOOMS.items()):
    axz = fig.add_subplot(gr[i])
    x0, y0, x1, y1 = boxes[k]
    set_view(axz, x0, x1, y0, y1)
    ctx.add_basemap(axz, source=BASEMAP, crs='EPSG:3857', attribution=False)
    sub = s[s['zoom'] == k]
    draw_points(axz, sub, sub, all_pts, size=95, fs=8.5)
    axz.set_xticks([]); axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_edgecolor('white'); sp.set_linewidth(1.0)
    axz.set_title(f'{k}: {title} ({len(sub)} stations)', fontsize=9)

# ---- bảng số -> mã trạm ----
axt = fig.add_subplot(gr[-1]); axt.axis('off')
rows = [f'{r.num:>2}  {r.site_no:<9} {r.state}' for r in s.itertuples()]
half = (len(rows) + 1) // 2
for c, col in enumerate((rows[:half], rows[half:])):
    axt.text(0.02 + c * 0.5, 0.93, '\n'.join(col), va='top', ha='left', family='monospace',
             fontsize=7.2, linespacing=1.3, transform=axt.transAxes)
axt.text(0.02, 0.97, 'No.  USGS site  State', fontsize=7.5, fontweight='bold',
         va='bottom', transform=axt.transAxes)

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('Đã lưu', OUT, '| số trạm:', len(s), '|', {k: int((s.zoom == k).sum()) for k in ZOOMS})
