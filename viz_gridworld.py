import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def create_grid_world(grid):

    colors = ['white', 'blue', 'red']
    cmap = ListedColormap(colors)


    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, cmap=cmap, fmt='d', cbar=False, square=True, linewidths=1, linecolor='black')

    # Add grid lines for boxes
    plt.imshow(grid, cmap=cmap, alpha=0, interpolation='nearest', extent=[0, 11, 0, 11], origin='lower', aspect='equal')
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=[0, 1, 2], label='Player 1 (Blue) vs Player 2 (Red)')
    plt.title('Connect 4')
    plt.show()
