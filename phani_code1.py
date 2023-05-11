coding: utf-8 -*-
"""
Created on Fri May 12 02:24:11 2023

@author: PHANINDRA SAI NAIDU
"""

# -- coding: utf-8 --

#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt

#Used from class
import cluster_tools as ct
import errors as err
import importlib


def data_reading(filename):
    '''
    data_reading will create dataframe of gven filename

    Parameters
    ----------
    filename : STR
        File path or location.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame created from given filepath.

    '''
    df = pd.read_csv(filename, skiprows=4)
    df = df.set_index('Country Name', drop=True)
    df = df.loc[:, '1960':'2021']

    return df


def data_transpose(df):
    '''
    data_transpose creates an new dataframe, transpose of given dataframe

    Parameters
    ----------
    df  : pandas.DataFrame
        DataFrame for which transpose to be found.

    Returns
    -------
    data_tr : pandas.DataFrame
        Transposed DataFrame of given DataFrame.

    '''
    df_tr = df.transpose()

    return df_tr


def correlation_and_scattermatrix(df):
    '''
    correlation_and_scattermatrix plots correlation matrix and scatter plots
    of data among columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame for which analysis will be done.

    Returns
    -------
    None.

    '''
    corr = df.corr()
    print(corr)
    plt.figure(figsize=(10, 10))
    plt.matshow(corr, cmap='coolwarm')

    # xticks and yticks for corr matrix
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(
        'Electricity power consumption kWh per capita')
    plt.colorbar()
    plt.show()

    pd.plotting.scatter_matrix(df, figsize=(12, 12), s=5, alpha=0.8)
    plt.show()

    return


def cluster_number(df, df_normalised):
    '''
    cluster_number calculates the best number of clusters based on silhouette
    score

    Parameters
    ----------
    df : pandas.DataFrame
        Actual data.
    df_normalised : pandas.DataFrame
        Normalised data.

    Returns
    -------
    INT
        Best cluster number.

    '''

    clusters = []
    scores = []
    # loop over number of clusters
    for ncluster in range(2, 10):

        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Cluster fitting
        kmeans.fit(df_normalised)
        lab = kmeans.labels_

        # Silhoutte score over number of clusters
        print(ncluster, skmet.silhouette_score(df, lab))

        clusters.append(ncluster)
        scores.append(skmet.silhouette_score(df, lab))

    clusters = np.array(clusters)
    scores = np.array(scores)

    best_ncluster = clusters[scores == np.max(scores)]
    print()
    print('best n clusters', best_ncluster[0])

    return best_ncluster[0]


def clusters_and_centers(df, ncluster, y1, y2):
    '''
    clusters_and_centers will plot clusters and its centers for given data

    Parameters
    ----------
    df : pandas.DataFrame
        Data for which clusters and centers will be plotted.
    ncluster : INT
        Number of clusters.
    y1 : INT
        Column 1
    y2 : INT
        Column 2

    Returns
    -------
    df : pandas.DataFrame
        Data with cluster labels column added.
    cen : array
        Cluster Centers.

    '''
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df)

    labels = kmeans.labels_
    df['labels'] = labels
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    cen = np.array(cen)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    sc = plt.scatter(df[y1], df[y2], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(f"Electricity power consumption per capita kWh({y1})")
    plt.ylabel(f"Electricity power consumption per capita kWh({y2})")
    plt.legend(*sc.legend_elements(), title='clusters')
    plt.title('Electricity power consumption kWh per capita in 1970 and 2020')
    plt.show()

    print()
    print(cen)

    return df, cen


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 
    and growth rate g"""

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f
