import numpy as np
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN

class CustomScenario:
    def __init__(self, _features=['Cellular Subscription', 'Internet Users(%)', 'No. of Internet Users', 'Broadband Subscription']):
        self._features = _features

    def visualize_scenario_func(self,
                                 df_decade_v1,
                                 clustering_method = 'K-Means', 
                                 pca_components = 2, 
                                 clusters_num = 5, 
                                 kmeans_init_method = 1,
                                 dbscan_params = [[0.15, 5]], 
                                 dbscan_chosen_params = 0, 
                                 show_corr = False):
        """
        The main function for executing a scenario of dimensionality reduction and clustering, followed by visualization on graphs.

        Parameters
        ----------
        df_decade_v1 : pd.DataFrame
            Initial version of the unscaled dataset. 
        clustering_method : str
            Choice of clustering method: 'K-Means' or 'DBSCAN' (optional). Default is 'K-Means'.
        pca_components : int
            Number of components (2 or 3) for PCA dimensionality reduction. Default is 2.
        clusters_num : int
            Number of clusters for K-Means clustering. Default is 5.
        kmeans_init_method : int
            Choice of initialization method for initial points in K-Means clustering. Default is 1 ('k-means++').
        dbscan_params : array_like[array_like]
            List of hyperparameter pairs (epsilon and min_samples) for DBSCAN clustering. Default is [[0.5, 5]].
        dbscan_chosen_params : int
            Index of the chosen hyperparameter pair from dbscan_params for DBSCAN clustering. Default is 0.
        show_corr : bool
            Display correlation matrix for PCA algorithm. Default is False.

        Notes
        ----------
        The order of algorithms in a possible scenario is as follows:
        1. Data Scaling
            1.1. scale_data_func
        2. PCA
            2.1. PCA_func (2 or 3 components)
            2.2. visualize_corr_for_PCA_func - optional
        3. Clustering
            3.1. KMeans_func (N clusters)
            3.2. DBSCAN_func - optional
        4. t-SNE - optional
            4.1. TSNE_func (N clusters)
            4.2. visualize_TSNE_func


        You can customize the order of algorithm execution according to your preferences. 
        You are also free to execute intermediate functions without calling the main visualize_scenario_func 
        in the desired order. However, executing the two main functions: scale_data_func and PCA_func, 
        is necessary for correct clustering and convenient visualization of results. 
        """
        
        # Necessary functions
        scale_data = self.scale_data_func(df_decade_v1)
        PCA_data = self.PCA_func(scale_data, components_num = pca_components)
        
        # Selection of clustering method
        if clustering_method == 'K-Means':
            df_decade_v2 = self.KMeans_func(df_decade_v1, PCA_data, clusters_num, kmeans_init_method)
        elif clustering_method == 'DBSCAN':
            df_decade_v2 = self.DBSCAN_func(df_decade_v1, PCA_data, dbscan_params, dbscan_chosen_params)
        else:
            print('N/A clustering method')
        
        # Visualization of the corr matrix
        if show_corr:
            self.visualize_corr_for_PCA_func(scale_data, PCA_data, pca_components)
            
        # Visualize the clusters
        if pca_components == 2:
            fig = px.scatter(df_decade_v2, x='PCA_1', y='PCA_2', color='Cluster', 
                            hover_data=['Entity'], title=f'{clustering_method} clusters for 2020')
            fig.show()
        elif pca_components == 3:
            fig = px.scatter_3d(df_decade_v2,
                        x = 'PCA_1', 
                        y = 'PCA_2',
                        z = 'PCA_3', 
                        color='Cluster', 
                        hover_data=['Entity'])
            fig.show()
        else:
            print('Too many dimensions for visualize!')
            
    def scale_data_func(self, df_decade_v1):
        df_decade_v1[self._features].fillna(df_decade_v1[self._features].mean(), inplace=True)
        
        # Standardize the features
        scaler = StandardScaler()
        df_decade_scaled = df_decade_v1.copy()
        df_decade_scaled[self._features] = scaler.fit_transform(df_decade_v1[self._features])
        
        return df_decade_scaled[self._features]

    def PCA_func(self, df_decade_scaled, components_num = 2):
        # Perform PCA for dimensionality reduction
        model = PCA(n_components=components_num)
        X_pca = model.fit_transform(df_decade_scaled[self._features])
        
        pca_cols = [f'PCA_{i}' for i in range(1, components_num + 1)]
        df_decade_pca = pd.DataFrame(data = X_pca, columns = pca_cols)
        
        return df_decade_pca

    def TSNE_func(self, df_decade_scaled, df_decade_v1, Y_cluster, components_num = 2):
        model = TSNE(n_components=components_num)
        X_tsne = model.fit_transform(df_decade_scaled[self._features])
        
        X_tsne = np.concatenate((X_tsne, Y_cluster.values.reshape(-1, 1)), axis = 1)
        
        tsne_cols = [f'TSNE_{i}' for i in range(1, components_num + 1)]
        df = pd.DataFrame(X_tsne, columns=tsne_cols + ['Cluster'])
        
        df['Entity'] = df_decade_v1['Entity'].values
        
        return df

    def visualize_TSNE_func(self, df_decade_scaled, df_decade_v1, df_decade_v2, components_num = 2):
        df_TSNE = self.TSNE_func(df_decade_scaled, df_decade_v1, df_decade_v2['Cluster'], components_num)
        if components_num == 2:
            fig = px.scatter(df_TSNE, 
                            x='TSNE_1', 
                            y='TSNE_2', 
                            color='Cluster', 
                            hover_data=['Entity'], 
                            title=f'TSNE clusters for 2020')
            fig.show()
        elif components_num == 3:
            fig = px.scatter_3d(df_TSNE,
                            x = 'TSNE_1', 
                            y = 'TSNE_2',
                            z = 'TSNE_3', 
                            color='Cluster', 
                            hover_data=['Entity'])
            fig.show()
        else:
            print('Too many dimensions for visualize!')


    def KMeans_func(self, df_decade_v1, df_decade_v2, clusters_num = 5, init_method = 1):
        inits = ['random',
                'k-means++',
                np.array([[1, 1], [0, 0]]),
                np.array([[1, -2],[1, -2]])
            ]
        
        # Perform KMeans clustering
        model = KMeans(n_clusters=clusters_num, init = inits[init_method])
        clusters = model.fit_predict(df_decade_v2)
        df_decade_v2['Cluster'] = clusters
        df_decade_v2['Entity'] = df_decade_v1['Entity'].values
        
        return df_decade_v2
        
    def DBSCAN_func(self, df_decade_v1, df_decade_v2, params = [[0.15, 5]], chosen_params = 0):
        '''Optional'''
        model = DBSCAN(eps=params[chosen_params][0], min_samples=params[chosen_params][1])
        clusters = model.fit_predict(df_decade_v2)
        df_decade_v2['Cluster'] = clusters
        df_decade_v2['Entity'] = df_decade_v1['Entity'].values
        
        return df_decade_v2


    def visualize_corr_for_PCA_func(self, scale_data, PCA_data, components_num = 2):
        '''Optional'''
        if components_num == 2:
            first_component_corr_2d = scale_data.reset_index().corrwith(PCA_data.reset_index().PCA_1)
            second_component_corr_2d = scale_data.reset_index().corrwith(PCA_data.reset_index().PCA_2)
            
            corrs = pd.concat((first_component_corr_2d, second_component_corr_2d), axis=1)
            corrs.columns = ['PCA_1', 'PCA_2']
            
        elif components_num == 3:
            first_component_corr_3d = scale_data.reset_index().corrwith(PCA_data.reset_index().PCA_1)
            second_component_corr_3d = scale_data.reset_index().corrwith(PCA_data.reset_index().PCA_2)
            third_component_corr_3d = scale_data.reset_index().corrwith(PCA_data.reset_index().PCA_3)
            
            corrs = pd.concat((first_component_corr_3d, 
                            second_component_corr_3d, 
                            third_component_corr_3d), axis=1)
            corrs.columns = ['PCA_1', 'PCA_2', 'PCA_3']
        
        else:
            print('Too many dimensions for visualize!')
            
        corrs.drop('index', axis = 0, inplace = True)
        
        fig = plt.figure()
        fig.set_size_inches(16, 10)
        sns.heatmap(corrs, 
                    xticklabels=corrs.columns,
                    yticklabels=corrs.index,
                    cmap='BrBG',
                    vmin=-1,
                    vmax=1)

        plt.show()

