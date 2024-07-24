import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv("Data/labeled_db.csv", parse_dates=["Timestamp"])

# Define the desired order of features within each bar group
desired_feature_order = ['CPU-usage', 'Interrupts-per-sec',
                         'Num-processes', 'RAM-percentage', 'DSK-write', 'DSK-read']

# Ensure the selected features are present in the dataset
features = [feature for feature in desired_feature_order if feature in df.columns]

# Normalize values to be in the range of 0 to 100%
df[features] = df[features].apply(lambda x: (
    x - x.min()) / (x.max() - x.min()) * 100)

# Define consistent colors for each feature
feature_colors = plt.cm.tab10.colors[:len(features)]


SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

# Create a grouped bar chart for each feature
fig, ax = plt.subplots(figsize=(10, 8))

bar_width = 0.15
positions = np.arange(len(df['Attack-type'].unique()))


# Iterate over attack types and plot each one side by side for each feature
for i, attack_type in enumerate(df['Attack-type'].unique()):
    attack_data = df[df['Attack-type'] == attack_type][features]

    # Plot bars for each feature with consistent colors and desired order
    bars = ax.bar(i + np.arange(len(features)) * bar_width,
                  attack_data[desired_feature_order].mean(), width=bar_width, color=feature_colors, alpha=0.7)

    # Add markers for the 25th and 75th percentiles
    ax.errorbar(i + np.arange(len(features)) * bar_width + bar_width / 2,
                attack_data[desired_feature_order].median(),
                yerr=[attack_data[feature].median() - attack_data[feature].quantile(0.25)
                      for feature in desired_feature_order],
                fmt='o', color='black', markersize=8, capsize=5, label='_nolegend_')

# Create a legend with consistent colors and feature labels
ax.legend(bars, desired_feature_order,
          loc='upper left')

ax.set_ylabel('Percentage of Max Recorded Feature Values')
ax.set_ylim(0, 100)  # Set Y-axis limit to 0-100%
plt.xlabel('Traffic Type')
# plt.xticks(np.arange(len(df['Attack-type'].unique())) + (bar_width * (len(features) - 1) / 2), [
#            attack_type_mapping.get(att_type.lower(), att_type.title()) for att_type in df['Attack-type'].unique()])

plt.xticks(np.arange(len(df['Attack-type'].unique())) + (bar_width * (
    len(features) - 1) / 2), [att_type for att_type in df['Attack-type'].unique()])

plt.xticks(rotation=45, ha='right')
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the subplot layout
fig.savefig('figure/ddos-sys.png')
plt.show()
