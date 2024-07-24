import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Load the labeled dataset
labeled_file = 'Data/labeled_db.csv'
df_labeled = pd.read_csv(labeled_file, parse_dates=['Timestamp'])

# Exclude non-numeric columns (e.g., 'Attack_Type') for correlation analysis
numeric_columns = df_labeled.select_dtypes(
    include=['float64', 'int64']).columns
correlation_matrix = df_labeled[numeric_columns].corr(
    method='pearson')  # Use Pearson correlation

# Create a 2x1 grid layout
fig = plt.figure(figsize=(8, 8))
# Setting the font to Calibri
# plt.rcParams['font.family'] = 'Calibri'

gs = gridspec.GridSpec(2, 1, height_ratios=[0.1, 0.9])

# Plot the correlation matrix using Seaborn without numeric values
heatmap_ax = plt.subplot(gs[1])
heatmap = sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f", linewidths=.5, xticklabels=correlation_matrix.columns,
                      yticklabels=correlation_matrix.columns, cbar_kws={'label': 'Correlation Coefficient'}, annot=False, ax=heatmap_ax)

# Increase the font size of X and Y-axis labels
heatmap.set_xticklabels(heatmap.get_xticklabels(), size=12)
# Rotate Y-axis labels to the right
heatmap.set_yticklabels(heatmap.get_yticklabels(), size=12,  ha='right')

# Rotate X-axis labels to the right
plt.xticks(ha='right', rotation=45)

# Adjust figure layout to prevent label cutoff
plt.subplots_adjust(bottom=0.17)

# Add description above the matrix
description_ax = plt.subplot(gs[0])
description_ax.text(0.5, 0.5, '', ha='center', va='center',
                    fontsize=16)  # Empty string for no title
description_ax.axis('off')
# tight layout
plt.tight_layout()
fig.savefig('figure/ddos-corr.png')
plt.show()
