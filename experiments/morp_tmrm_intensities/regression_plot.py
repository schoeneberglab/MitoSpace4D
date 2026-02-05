import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm'
embeddings_dir = '/home/earkfeld/Projects/MitoSpace4D/runs/20260116_kinetics-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm'

tmrm_data = np.load(f'{embeddings_dir}/tmrm_intensities.npy')
morph_data = np.load(f'{embeddings_dir}/morph_intensities.npy')

tmrm_data = np.mean(tmrm_data, axis=1)
morph_data = np.mean(morph_data, axis=1)

# Normalize intensities
tmrm_data = (tmrm_data - tmrm_data.min()) / (tmrm_data.max() - tmrm_data.min())
morph_data = (morph_data - morph_data.min()) / (morph_data.max() - morph_data.min())

# Calculate R2
r2 = np.corrcoef(tmrm_data, morph_data)[0, 1]**2

# Plot as a regression plot with TMRM on the x-axis and morphology on the y-axis
sns.regplot(x=tmrm_data, y=morph_data, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.xlabel('TMRM Intensity')
plt.ylabel('Morphology Intensity')
plt.title(f'Kinetics Morphology Intensity vs. TMRM Intensity (R²: {r2:.4f})')
plt.show()