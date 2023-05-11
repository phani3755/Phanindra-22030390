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
    plt.figure(figsize=(10, 10))

    cm = plt.cm.get_cmap('tab10')
    sc = plt.scatter(df[y1], df[y2], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(f"Electricity power consumption per capita kWh({y1})")
    plt.ylabel(f"Electricity power consumption per capita kWh({y2})")
    plt.legend(*sc.legend_elements(), title='Clusters')
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


def forecast_elec_pwr_cons_pc(df, country, start_year, end_year):
    '''
    forecast_Electricity power consumption kWh will analyse data and optimize 
    to forecast Electricity power consumption kWh of selected country

    Parameters
    ----------
    df : pandas.DataFrame
        Data for which forecasting analysis is performed.
    country : STR
        Country for which forecasting is performed.
    start_year : INT
        Starting year for forecasting.
    end_year : INT
        Ending year for forecasting.

    Returns
    -------
    None.

    '''
    df = df.loc[:, country]
    df = df.dropna(axis=0)

    df_elec_pwr_cons_pc = pd.DataFrame()

    df_elec_pwr_cons_pc['Year'] = pd.DataFrame(df.index)
    df_elec_pwr_cons_pc['Elec_Cons_kWh'] = pd.DataFrame(df.values)
    df_elec_pwr_cons_pc["Year"] = pd.to_numeric(df_elec_pwr_cons_pc["Year"])
    importlib.reload(opt)

    param, covar = opt.curve_fit(logistic, df_elec_pwr_cons_pc["Year"],
                                 df_elec_pwr_cons_pc["Elec_Cons_kWh"],
                                 p0=(1.2e12, 0.03, 1990.0))

    sigma = np.sqrt(np.diag(covar))

    year = np.arange(start_year, end_year)
    forecast = logistic(year, *param)
    low, up = err.err_ranges(year, logistic, param, sigma)
    plt.figure()
    plt.plot(df_elec_pwr_cons_pc["Year"], df_elec_pwr_cons_pc["Elec_Cons_kWh"],
             label="Electricity power consumption kWh")
    plt.plot(year, forecast, label="Forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("Electricity power consumption kWh per capita")
    plt.legend()
    plt.title(f'Electric power consumption per capita forecast for {country}')
    plt.savefig(f'{country}.png', bbox_inches='tight', dpi=300)
    plt.show()

    x = logistic(2030, *param)/1e9

    low, up = err.err_ranges(2030, logistic, param, sigma)
    sig = np.abs(up-low)/(2.0 * 1e9)
    print()
    print("Electricity power consumption kWh 2030", x*1e9, "+/-", sig*1e9)


#Reading Electricity power consumption kWh per capita Data
df_elec_pwr_cons_pc = data_reading("elec_power_cons_kwh_per_capita.csv")
print(df_elec_pwr_cons_pc.describe())

#Finding transpose of Electricity power consumption kWh per capita Data
df_elec_pwr_cons_pc_tr = data_transpose(df_elec_pwr_cons_pc)
print(df_elec_pwr_cons_pc_tr.head())

#Selecting years for which correlation is done for further analysis
df_epc = df_elec_pwr_cons_pc[["1990", '1995', "2000", '2005', "2010", '2014']]
print(df_epc.describe())

correlation_and_scattermatrix(df_epc)
year1 = "1990"
year2 = "2014"

# Extracting columns for clustering
df_ex = df_epc[[year1, year2]]
df_ex = df_ex.dropna(axis=0)

# Normalising data and storing minimum and maximum
df_norm, df_min, df_max = ct.scaler(df_ex)

print()
print("Number of Clusters and Scores")
ncluster = cluster_number(df_ex, df_norm)

df_norm, cen = clusters_and_centers(df_norm, ncluster, year1, year2)

#Applying backscaling to get actual cluster centers
scen = ct.backscale(cen, df_min, df_max)
print('scen\n', scen)

df_ex, scen = clusters_and_centers(df_ex, ncluster, year1, year2)

'''
We can see some difference in actual cluster centers and 
backscaled cluster centers.
'''

print()
print('Countries in last cluster')
print(df_ex[df_ex['labels'] == ncluster-1].index.values)

#Forecast Electricity power consumption kWh per capita for Sweden
forecast_elec_pwr_cons_pc(df_elec_pwr_cons_pc_tr, 'Sweden', 1960, 2031)

#Forecast Electricity power consumption kWh per capita for Luxembourg
forecast_elec_pwr_cons_pc(df_elec_pwr_cons_pc_tr, 'Luxembourg', 1960, 2031)

#Forecast Electricity power consumption kWh per capita for Pakistan
forecast_elec_pwr_cons_pc(df_elec_pwr_cons_pc_tr, 'Pakistan', 1970, 2031)

#Forecast Electricity power consumption kWh per capita for Belgium
forecast_elec_pwr_cons_pc(df_elec_pwr_cons_pc_tr, 'Belgium', 1960, 2031)
