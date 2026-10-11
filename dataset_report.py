import config as C
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')

df = pd.read_csv(C.MANIFEST, dtype={'class': str})
summary = df.groupby(['split', 'class']).size().reset_index(name='image_count')
summary['low_support_warning'] = summary['image_count'] < 5
C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(C.RESULTS_DIR / 'dataset_summary.csv', index=False)

table = summary.pivot(index='class', columns='split',
                      values='image_count')[list(C.SPLITS)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, log in zip(axes, (False, True)):
    table.plot.bar(ax=ax, logy=log, rot=0)
    ax.set_title('Images per class' + (' (log scale)' if log else ''))
    ax.set_xlabel('Defect class')
    ax.set_ylabel('Number of images')
plt.tight_layout()
plt.savefig(C.RESULTS_DIR / 'class_distribution.png')
plt.close()
print(table)
