import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.cramers_v import cramers_v

def plot_cramer(df, vars):
    
    dataframe = pd.DataFrame()
    for x in vars:
        for y in vars:
            dataframe.loc[x, y] = cramers_v(df[x], df[y])
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(dataframe, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize = (11, 6))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap = True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataframe, mask = None, cmap = cmap, vmax = 1.0, center = 0,
                square = True, linewidths = .5, cbar_kws = {"shrink": .5})
    plt.title("Heatmap using Cramer's V")
    plt.show()