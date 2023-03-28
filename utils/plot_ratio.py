import matplotlib.pyplot as plt

def plot_ratio(df, feature, title, xlabel):
    
    '''
    Plots the delay ratio distributed by some feature.
    feature: name of the feature to plot against (str)
    title: plot title (str)
    xlabel: plot xlabel (str)
    '''
    
    plt.figure(figsize = (12, 4))
    (df.groupby(feature).mean()['delay_15'].sort_values(ascending = False) * 100).plot(kind = 'bar', label = 'Delay ratio')
    plt.axhline(df['delay_15'].mean()*100, color = 'r', label = 'Delay average', linestyle = '--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Delay Ratio (%)')
    plt.legend()