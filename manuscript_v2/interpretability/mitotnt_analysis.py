import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ranksums


def significance_test(sample1, sample2):
    # sample1 and sample2 are your two 1-D arrays of observations
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    if np.std(sample1) == 0 or np.std(sample2) == 0:
        print("One of the samples has zero variance, cannot perform t-test.")
        return None

    t_stat, p_val = ranksums(sample1, sample2, alternative='two-sided')  # Welch’s t-test by default
    print(f"p = {p_val:.3g}")


if __name__ == "__main__":
    mitotnt_features = pd.read_csv('/home/dhruvagarwal/projects/MitoSpace4D/data/adaptor_data/4d_mito_features.csv')

    oligo_feats = mitotnt_features[mitotnt_features['Date'] == 20240809][:300]
    h2o2_feats = mitotnt_features[mitotnt_features['Date'] == 20240805][:300]

    feature_name = 'Fusion Rate'

    print(significance_test(list(oligo_feats[feature_name]), list(h2o2_feats[feature_name])))

    oligo_feats_tmrm_intensity = oligo_feats[feature_name]
    h2o2_feats_tmrm_intensity = h2o2_feats[feature_name]

    feature_to_plot = [feature_name]

    # Combine the data for both conditions
    combined_feats = pd.concat([oligo_feats, h2o2_feats])
    combined_feats['Condition'] = combined_feats['Date'].map({20240809: 'Oligomycin', 20240805: 'H2O2'})

    # Define custom RGB colors
    # custom_palette = {'Control': (0, 0.501, 0.0),  # Example RGB (normalized to 0-1)
    #                   'P110': (0.439, 0.501, 0.564)}        # Example RGB
    custom_palette = {'Oligomycin': (0, 0.501, 0.0),  # Example RGB (normalized to 0-1)
                      'H2O2': (0.439, 0.501, 0.564)}

    # Plot the boxplot
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=combined_feats, x='Condition', y=feature_to_plot[0], palette=custom_palette, showfliers=False)
    plt.ylabel(feature_to_plot[0])
    plt.show()
