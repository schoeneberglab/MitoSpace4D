import numpy as np
from scipy.spatial import distance
import pandas as pd


def calculate_mahalanobis_distance(experimental_data, control_distribution):
    """
    Calculates the Mahalanobis distance for experimental samples against a control distribution.

    Args:
        experimental_data (np.array): A (n_samples x n_features) array of the experimental fold changes.
        control_distribution (np.array): A (n_samples x n_features) array of the control group data
                                         used to define the covariance matrix.

    Returns:
        np.array: An array of Mahalanobis distances for each experimental sample.
    """

    # 1. Calculate the mean of the control distribution
    # This represents the "center" of the multivariate space
    control_mean = np.mean(control_distribution, axis=0)

    # 2. Calculate the Covariance Matrix of the control distribution
    # rowvar=False indicates that rows are observations (samples) and columns are variables (genes/features)
    cov_matrix = np.cov(control_distribution, rowvar=False)

    # 3. Calculate the Inverse Covariance Matrix
    # We add a tiny value (regularization) to the diagonal to prevent singular matrix errors
    # if features are highly correlated or N < P.
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix detected. Adding regularization.")
        reg_cov = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
        inv_cov_matrix = np.linalg.inv(reg_cov)

    # 4. Calculate Distances
    distances = []
    for sample in experimental_data:
        # distance.mahalanobis expects 1D arrays for u and v
        d = distance.mahalanobis(sample, control_mean, inv_cov_matrix)
        distances.append(d)

    return np.array(distances)


import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import chi2


def analyze_mahalanobis(controls, experimental):
    """
    Computes Mahalanobis distance and P-values for an experimental set
    based on the control set's distribution.

    Args:
        controls (np.array): (n_control_samples x n_features)
        experimental (np.array): (n_exp_samples x n_features)

    Returns:
        pd.DataFrame: DataFrame containing distances and p-values for experimental samples.
    """

    # 1. Define the 'Normal' Space using Controls
    # We use the control group to define what "baseline" correlations look like.
    control_mean = np.mean(controls, axis=0)

    # Calculate Covariance using the Control group only
    # rowvar=False means rows are samples, columns are features
    cov_matrix = np.cov(controls, rowvar=False)

    # 2. Invert Covariance Matrix (Precision Matrix)
    # Includes a safety check for singular matrices (common in high-dimensional data)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Matrix is singular. Adding regularization (pseudo-inverse).")
        # Use pseudo-inverse if strict inversion fails
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # 3. Calculate Distances for Experimental Samples
    results = []

    for i, sample in enumerate(experimental):
        # Calculate Mahalanobis distance
        d_m = distance.mahalanobis(sample, control_mean, inv_cov_matrix)

        # Calculate Squared Mahalanobis distance (D^2) for Chi-Square test
        d_m_sq = d_m ** 2

        # Calculate P-value
        # Degrees of freedom (df) = number of features (genes/metrics)
        df = controls.shape[1]
        p_value = 1 - chi2.cdf(d_m_sq, df)

        results.append({
            "Sample_Index": i,
            "Mahalanobis_Dist": d_m,
            "Mahalanobis_Sq": d_m_sq,
            "P_Value": p_value
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260117_2024v2-raw_kinetics-resnet3d_ablated_tmrm_extract_tmrm"
    embeddings_dir = "/home/earkfeld/Projects/MitoSpace4D/runs/20260121_liver-drugs_4D-embeddings_2024v2-model"

    embeddings = np.load(f"{embeddings_dir}/embeddings_raw.npy")
    labels = np.load(f"{embeddings_dir}/labels.npy")
    label_names = np.load(f"{embeddings_dir}/label_names.npy")

    unique_labels = np.unique(labels)
    experiment_labels = [label for label in unique_labels if label != 0]  # Assuming '0' is control
    print("Unique Labels:", unique_labels)
    print("Experiment Labels (non-control):", experiment_labels)

    control_idxs = np.where(labels == 0)[0]  # Assuming label '0' corresponds to control
    control_group = embeddings[control_idxs]

    experiment_groups = {}
    for label in experiment_labels:
        experiment_idxs = np.where(labels == label)[0]
        experiment_groups[label] = embeddings[experiment_idxs]

    # V1
    # # Calculate Mahalanobis Distances for each experimental group
    # results = {}
    # for label, exp_data in experiment_groups.items():
    #     mahalanobis_dists = calculate_mahalanobis_distance(exp_data, control_group)
    #     results[label] = mahalanobis_dists
    #
    # # Display Results
    # for label, dists in results.items():
    #     print(f"\n--- Results for Label {label} ({label_names[label]}) ---")
    #     print(f"Mahalanobis Distances: {dists}")

    # V2
    # Analyze Mahalanobis distances and p-values for each experimental group
    all_results = []
    for label, exp_data in experiment_groups.items():
        df_results = analyze_mahalanobis(control_group, exp_data)
        df_results["Label"] = label
        df_results["Label_Name"] = label_names[label]
        # all_results.append(df_results)

        # Calculate the fold-change distance from control
