import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_agreement_heatmap(anno1, anno2, block_sizes, block_labels, colors=None):
    """
    Plots a tall heatmap for two annotators with range markers on the left.
    
    Args:
        anno1, anno2 (np.array): Arrays of length N (e.g., 200).
        block_sizes (list): List of 3 integers summing to N.
        block_labels (list): Names for the three blocks.
    """
    data = np.stack([anno1, anno2], axis=1)
    n_rows = len(anno1)
    
    # Create figure with a specific aspect ratio for a tall heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Display heatmap
    im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Formatting columns
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Annotator 1', 'Annotator 2'])
    ax.set_yticks([]) # Hide individual row labels
    
    # Draw horizontal lines and range indicators
    current_row = 0
#     colors = ['#FF5733', '#33FF57', '#3357FF'] # Distinct colours for range lines
    if not colors:
        colors = ['k'] * len(block_sizes)  # Distinct colours for range lines
    
    for i, size in enumerate(block_sizes):
        start = current_row
        end = current_row + size
        mid = (start + end) / 2
        
#         # Draw a horizontal line separating blocks on the heatmap itself
#         if end < n_rows:
#             ax.axhline(end - 0.5, color='white', linewidth=1)
            
        # Draw a bracket-like line to the left of the y-axis
        # Transform coords to axes fraction for the 'offset' look
        ax.annotate('', xy=(-0.05, start), xycoords=('axes fraction', 'data'),
                    xytext=(-0.05, end), textcoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-', color=colors[i], linewidth=3))
        
        # Add text for the block label
        ax.text(-0.1, mid, block_labels[i], transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontweight='bold', color=colors[i])
        
        current_row = end

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correct/Incorrect')
    plt.tight_layout()
    
    
def plot_latent_facility_ec_experiment(
        df_sim,
        variance,
        figsize=(10, 6)
):
    plt.figure(figsize=figsize)
    plt.scatter(df_sim['mean_accuracy'], df_sim['kappa'], alpha=0.3, s=10, label='Trials')
    # Group by mean_accuracy and calculate the mean kappa for each group
    trend = df_sim.groupby('mean_accuracy')['kappa'].mean()

    # Convert the index and values to numpy arrays to avoid the indexing error
    x_vals = trend.index.to_numpy()
    y_vals = trend.values # .values is already an array, but .to_numpy() is safer

    plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Trend Line')
    plt.xlabel('Mean Item Facility ($\mu$)')
    plt.ylabel('Error Consistency ($\kappa$)')
    plt.title(
        f'Effect of Accuracy on EC with latent facility variance ({variance})')
    plt.legend()

