import matplotlib.pyplot as plt
import numpy as np



def bar_plot(x,y, title=None, x_label=None,y_label=None, size=None, color=None, val_sort=False, plot_x_ticks=True):
    
    x = np.array(x)
    y = np.array(y)

    if val_sort == True:
        indices_to_sort_by = np.argsort(y)[::-1] ##Basically lets you store the indices x would be sorted by
    else:
        indices_to_sort_by = np.argsort(x)

    x_sorted = x[indices_to_sort_by] ##We then use these sorted indices, to make sure when we sort x, we don't lose it's relationship to y
    y_sorted = y[indices_to_sort_by]


    fig = plt.figure()
    #plt.clf()

    xrng = np.arange(len(x))

    plt.bar(xrng, y_sorted, 0.45, color=color)

    if size is not None:
        fig.set_size_inches(size)
    if plot_x_ticks:
        if len(x) > 4:
            plt.xticks(xrng, x_sorted, rotation=45, ha='right')
        else:
            plt.xticks(xrng, x_sorted)        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    pass

def hist_plot(x, title=None, x_label=None,y_label=None, size=None, color=None, bin_count=None):

    fig = plt.figure()
    plt.clf()
    plt.hist(x, bins=bin_count, color=color, edgecolor = "black")

    if size is not None:
        fig.set_size_inches(size)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    pass

def scatter_plot(x,y, title=None, x_label=None,y_label=None, size=None, color=None, point_size=None):
    fig = plt.figure()
    plt.clf()
    plt.scatter(x, y, color=color, s=point_size)

    if size is not None:
        fig.set_size_inches(size)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    pass

def box_plot(x,y, title=None, x_label=None,y_label=None, size=None):
    unique_xvals_dict = {}
    unique_xvals = list(set(x))

    for xval in unique_xvals:
        unique_xvals_dict[xval] = []

    for index in range(len(x)):
        unique_xvals_dict[x[index]].append(y[index])

    plt.boxplot(unique_xvals_dict.values())
    x_ticks = np.arange(len(unique_xvals_dict.values())) + 1
    plt.xticks(x_ticks,unique_xvals_dict.keys())

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()