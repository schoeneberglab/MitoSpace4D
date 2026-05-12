# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict
#
#
# def extract_rmae_data(base_path):
#     """
#     Extract RMAE data from all prediction CSV files in feature folders.
#
#     Args:
#         base_path (str): Base path to the feature folders
#
#     Returns:
#         dict: Dictionary with feature names as keys and lists of RMAE values as values
#     """
#     rmae_data = {}
#
#     # Walk through directories in the base path
#     for item in os.listdir(base_path):
#         feature_path = os.path.join(base_path, item)
#
#         # Check if it's a directory
#         if os.path.isdir(feature_path):
#             feature_name = os.path.basename(feature_path)
#             print(f"Processing feature: {feature_name}")
#
#             # Look for prediction CSV files
#             for file in os.listdir(feature_path):
#                 if file.endswith('_predictions.csv'):
#                     csv_path = os.path.join(feature_path, file)
#                     print(f"  Found predictions file: {file}")
#
#                     try:
#                         # Read the CSV file
#                         df = pd.read_csv(csv_path)
#
#                         # Check if RMAE column exists
#                         if 'RMAE' in df.columns:
#                             print(f"  RMAE column found in {file}")
#                             # Store the entire RMAE column as a list
#                             rmae_values = df['RMAE'].dropna().tolist()
#                             rmae_data[feature_name] = rmae_values
#                             print(f"  Extracted {len(rmae_values)} RMAE values")
#                         else:
#                             print(f"  No RMAE column found in {file}")
#                     except Exception as e:
#                         print(f"  Error processing {file}: {str(e)}")
#
#     return rmae_data
#
#
# def create_dataframe(rmae_data):
#     """
#     Create a DataFrame with feature names as column headers and RMAE values as contents.
#
#     Args:
#         rmae_data (dict): Dictionary with feature names as keys and lists of RMAE values
#
#     Returns:
#         DataFrame: pandas DataFrame with feature columns
#     """
#     # Determine the maximum length of RMAE lists
#     max_length = max(len(values) for values in rmae_data.values()) if rmae_data else 0
#
#     # Create a dictionary for the DataFrame with padded lists
#     df_dict = {}
#     for feature, values in rmae_data.items():
#         # Pad with NaN if necessary to ensure all columns have the same length
#         padded_values = values + [np.nan] * (max_length - len(values))
#         df_dict[feature] = padded_values
#
#     # Create the DataFrame
#     df = pd.DataFrame(df_dict)
#
#     return df
#
#
# def create_boxplot(df, output_path):
#     """
#     Create a boxplot of RMAE values by feature.
#
#     Args:
#         df (DataFrame): DataFrame with feature columns
#         output_path (str): Path to save the boxplot image
#     """
#     if df.empty:
#         print("No data available for creating boxplot")
#         return
#
#     plt.figure(figsize=(12, 8))
#
#     # Melt the DataFrame to long format for seaborn
#     melted_df = df.melt(var_name='Feature', value_name='RMAE')
#     melted_df = melted_df.dropna()  # Remove NaN values
#
#     # Create the boxplot
#     ax = sns.boxplot(x='Feature', y='RMAE', data=melted_df, showfliers=False, )
#     # ax = sns.boxplot(x='Feature', y='RMAE', data=melted_df)
#
#     # Customize the plot
#     plt.title('RMAE Distribution by Feature', fontsize=16)
#     plt.xlabel('Feature', fontsize=14)
#     plt.ylabel('RMAE Value', fontsize=14)
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#
#     # Save the plot
#     plt.savefig(output_path)
#     print(f"Boxplot saved to {output_path}")
#
#     # Display statistics
#     print("\nRMAE Statistics by Feature:")
#     print(df.describe())
#
#
# if __name__ == "__main__":
#     # Define paths (modify these as needed)
#     base_path = "/home/dhruvagarwal/projects/MitoSpace4D/adaptors/single_dataset/figures/regression_figures"  # Replace with actual path to your feature folders
#     output_plot = "rmae_boxplot.png"
#
#     # Extract RMAE data
#     rmae_data = extract_rmae_data(base_path)
#
#     # Create DataFrame with feature columns
#     rmae_df = create_dataframe(rmae_data)
#
#     # Print the DataFrame structure
#     print("\nDataFrame Structure:")
#     print(rmae_df.head())
#
#     # Create boxplot
#     create_boxplot(rmae_df, output_plot)

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def extract_rmae_data(base_path):
    """
    Extract RMAE data from all prediction CSV files in feature folders.

    Args:
        base_path (str): Base path to the feature folders

    Returns:
        dict: Dictionary with feature names as keys and lists of RMAE values as values
    """
    rmae_data = {}

    # Walk through directories in the base path
    for item in os.listdir(base_path):
        feature_path = os.path.join(base_path, item)

        # Check if it's a directory
        if os.path.isdir(feature_path):
            feature_name = os.path.basename(feature_path)
            if 'Std' in feature_name:
                continue
            print(f"Processing feature: {feature_name}")

            # Look for prediction CSV files
            for file in os.listdir(feature_path):
                if file.endswith('_predictions.csv'):
                    csv_path = os.path.join(feature_path, file)
                    print(f"  Found predictions file: {file}")

                    try:
                        # Read the CSV file
                        df = pd.read_csv(csv_path)

                        # Check if RMAE column exists
                        if 'RMAE' in df.columns:
                            print(f"  RMAE column found in {file}")
                            # Store the entire RMAE column as a list
                            rmae_values = df['RMAE'].dropna().tolist()
                            rmae_data[feature_name] = rmae_values
                            print(f"  Extracted {len(rmae_values)} RMAE values")
                        else:
                            print(f"  No RMAE column found in {file}")
                    except Exception as e:
                        print(f"  Error processing {file}: {str(e)}")

    return rmae_data


def create_boxplot(rmae_dict, output_path):
    """
    Create a visually appealing boxplot of RMAE values by feature.

    Args:
        rmae_dict (dict): Dictionary with feature names as keys and RMAE value lists as values
        output_path (str): Path to save the boxplot image
    """
    if not rmae_dict:
        print("No data available for creating boxplot")
        return

    plt.figure(figsize=(14, 8))  # Larger figure for readability

    # Convert dictionary to long format DataFrame
    feature_names, rmae_values = [], []
    for feature, values in rmae_dict.items():
        feature_names.extend([feature] * len(values))
        rmae_values.extend(values)

    rmae_values = [x * 100 for x in rmae_values]  # Convert to percentage
    df = pd.DataFrame({"Feature": feature_names, "RMAPE": rmae_values})

    # Use a cleaner style
    sns.set_style("whitegrid")

    # Create boxplot with enhanced visuals
    ax = sns.boxplot(
        x="Feature",
        y="RMAPE",
        data=df,
        showfliers=False,  # Hide outliers
        palette="Set2",  # Soft pastel colors
        linewidth=2,  # Thicker box lines
        boxprops=dict(edgecolor="black", linewidth=1.5),  # Black box edges
        medianprops=dict(color="red", linewidth=2),  # Red median line
        whiskerprops=dict(color="black", linewidth=1.5),  # Whiskers
    )

    # # Add jittered scatter points for individual values
    # sns.stripplot(
    #     x="Feature",
    #     y="RMAE",
    #     data=df,
    #     jitter=True,  # Spread points horizontally
    #     alpha=0.5,  # Semi-transparent
    #     color="black",  # Black dots
    #     size=4,  # Dot size
    # )

    # Customize title and labels
    plt.title("RMAPE Distribution by Feature", fontsize=18, fontweight="bold")
    plt.xlabel("Feature", fontsize=14, fontweight="bold")
    plt.ylabel("RMAPE (%)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12)  # Rotate x-labels
    plt.yticks(fontsize=12)

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"Boxplot saved to {output_path}")


if __name__ == "__main__":
    # Define paths (modify these as needed)
    base_path = "/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace/adaptors/figures/regression_figures"  # Replace with actual path to your feature folders
    output_plot = "/home/dhruvagarwal/projects/MitoSpace4D/mitodevXmitospace/adaptors/figures/rmae_boxplot.png"

    # Extract RMAE data to dictionary
    rmae_dict = extract_rmae_data(base_path)

    # Print the dictionary structure
    print("\nDictionary Structure:")
    for feature, values in rmae_dict.items():
        print(f"{feature}: {len(values)} values (first few: {values[:5]}...)")

    # Create boxplot
    create_boxplot(rmae_dict, output_plot)