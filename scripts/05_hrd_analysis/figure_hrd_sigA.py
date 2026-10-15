"""
Figure: CH295A Activity in HRD-Associated Cancer Types
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Config
ACTIVITIES_FILE = "CH295_activities_proportions.tsv"
OUTPUT_DIR = "Figure_HRD"

HRD_CANCER_TYPES = ['Ovary-AdenoCA', 'Breast-AdenoCA', 'Panc-AdenoCA', 'Prost-AdenoCA']

CANCER_COLORS = {
    'Ovary-AdenoCA': '#e74c3c',
    'Breast-AdenoCA': '#3498db',
    'Panc-AdenoCA': '#2ecc71',
    'Prost-AdenoCA': '#9b59b6'
}

# Load data
df = pd.read_csv(ACTIVITIES_FILE, sep='\t', index_col=0)
df['cancer_type'] = df.index.str.split('::').str[0]
df_hrd = df[df['cancer_type'].isin(HRD_CANCER_TYPES)].copy()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=df_hrd,
    x='cancer_type',
    y='CH295A',
    order=HRD_CANCER_TYPES,  # Force correct order
    hue='cancer_type',
    hue_order=HRD_CANCER_TYPES,
    palette=CANCER_COLORS,
    legend=False,
    ax=ax
)

sns.stripplot(
    data=df_hrd,
    x='cancer_type',
    y='CH295A',
    order=HRD_CANCER_TYPES,  # Force correct order
    color='black',
    alpha=0.4,
    size=4,
    ax=ax
)

# Add sample counts
for i, cancer in enumerate(HRD_CANCER_TYPES):
    n = len(df_hrd[df_hrd['cancer_type'] == cancer])
    mean_val = df_hrd[df_hrd['cancer_type'] == cancer]['CH295A'].mean()
    ax.text(i, 1.05, f'n={n}\nmean={mean_val:.2f}', ha='center', fontsize=9)

ax.set_xlabel('Cancer Type', fontsize=12)
ax.set_ylabel('CH295A Activity (Proportion)', fontsize=12)
ax.set_title('CH295A Signature Activity in HRD-Associated Cancer Types', fontsize=14, fontweight='bold')
ax.set_ylim(-0.05, 1.15)
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'fig_CH295A_hrd_boxplot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {output_path}")

# Print stats
print("\nCH295A Activity Summary:")
print("-" * 50)
for cancer in HRD_CANCER_TYPES:
    data = df_hrd[df_hrd['cancer_type'] == cancer]['CH295A']
    print(f"{cancer:20} n={len(data):3}  mean={data.mean():.3f}  median={data.median():.3f}")
