#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:00:59 2024

"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from prince import PCA as PCA_Prince
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from predictPy import Analisis_Predictivo
from sklearn.linear_model import LinearRegression
import ModMC
from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
class AnalisisDatosExploratorio():
    def __init__(self, path):
        self.__df = self.__cargarDatos(path)  
    #GETS
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
            
    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include = ["number"])
        self.__df = pd.DataFrame(StandardScaler().fit_transform(self.__df), columns= self.__df.columns)

    def analisisCompleto(self):  
        self.__df = pd.get_dummies(self.__df)
            
    def __cargarDatos(self, path):
        return pd.read_csv(path,
        sep = ",",
        decimal = ".",
        index_col = 0)
     
    def analisis(self): 
        print("Dimensiones:",self.__df.shape)
        print(self.__df.head)
        print(self.__df.describe())
        self.__df.dropna().describe()
        self.__df.mean(numeric_only=True)
        self.__df.median(numeric_only=True)
        self.__df.std(numeric_only=True, ddof = 0) 
        self.__df.max(numeric_only=True)
        self.__df.min(numeric_only=True)
        self.__df.quantile(np.array([0,.33,.50,.75,1]),numeric_only=True)

    def __str__(self):
        return f'AnalisisDatosExploratorio: {self.__df} {self.__path}'
  

class ACPBasico:
    def __init__(self, datos, n_componentes = 2): 
        self.__datos = datos
        self.__modelo = PCA_Prince(n_components = n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations
        self.__coordenadas_ind = self.__modelo.row_coordinates(self.__datos)
        self.__contribucion_ind = self.__modelo.row_contributions_
        self.__cos2_ind = self.__modelo.row_cosine_similarities(self.__datos)
        self.__var_explicada = self.__modelo.percentage_of_variance_
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    @property
    def modelo(self):
        return self.__modelo
    @property
    def correlacion_var(self):
        return self.__correlacion_var
    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind
    @property
    def contribucion_ind(self):
        return self.__contribucion_ind
    @property
    def cos2_ind(self):
        return self.__cos2_ind
    @property
    def var_explicada(self):
        return self.__var_explicada
    @var_explicada.setter
    def var_explicada(self, var_explicada):
        self.__var_explicada = var_explicada
    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo
    @correlacion_var.setter
    def correlacion_var(self, correlacion_var):
        self.__correlacion_var = correlacion_var
    @coordenadas_ind.setter
    def coordenadas_ind(self, coordenadas_ind):
        self.__coordenadas_ind = coordenadas_ind
    @contribucion_ind.setter
    def contribucion_ind(self, contribucion_ind):
        self.__contribucion_ind = contribucion_ind
    @cos2_ind.setter
    def cos2_ind(self, cos2_ind):
        self.__cos2_ind = cos2_ind
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')


class EDA(AnalisisDatosExploratorio):
    def __init__(self, df):
        self.__df = df 
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
        
    def drop(self, variables):
        self.__df.drop(variables, axis=1, inplace=True)
    def show(self, v):
        self.__df.info()
        self.__unique_values(v)
        self.__missing_values()
        self.__confusionmatrix()
        
        
    def __unique_values(self, v):
        unique_values = self.__df[v].unique()
    
        print("Valores únicos en", v,":")
        for value in unique_values:
            count = (self.__df[v] == value).sum()
            print(f"{value}: {count}")

    def __missing_values(self):
        missing_values = self.__df.isna().sum()

        print("Missing values by column:")
        print(missing_values)
        print('\n')
        

    def __confusionmatrix(self):
        correlation_matrix = self.__df.corr()
    
        color_start = np.array([224, 240, 246]) / 255
        color_end = np.array([1, 42, 74]) / 255
    
        num_segments = 100
        colors = [color_start + (color_end - color_start) * (i / num_segments) for i in range(num_segments)]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=num_segments)
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap=custom_cmap, fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
        
    def two_variable(self, v1, v2):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=v1, y=v2, data=self.__df, color='#014F86', alpha=0.5)
        plt.title(f'Plot of {v1} and {v2}')
        plt.xlabel(v1)
        plt.ylabel(v2)
        plt.show()
        
    def histogram(self, v):
        plt.hist(self.__df[v], color='#308AB0', edgecolor="black", bins=range(int(min(self.__df[v])), int(max(self.__df[v])) + 2, 1))
        plt.title("Histogram of " + v)
        plt.xlabel(v)
        plt.ylabel("Frequency")
        plt.show()

    def box(self, v1):
        fig = px.box(self.__df, y=v1, points="all", color_discrete_sequence=["#013A63"])
        fig.show()



class NoSupervisado(AnalisisDatosExploratorio):
    def __init__(self, df):
        self.__df = df
        self.__model = []
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
        
    @property
    def model(self):
        return self.__model
    @model.setter
    def model(self, p_model):
        self.__model = p_model
        
    def benchmark(self):
        df = pd.DataFrame(self.__model, columns=['Algoritmo', 'Número de Clusters', 'Silhouette Score'])
        return df
    
    def __agregar_modelo(self, algoritmo, n_clusters, silhouette_score):
        self.__model.append([algoritmo, n_clusters, silhouette_score])


    def ACP(self, n_componentes): 
        p_acp = ACPBasico(self.__df,n_componentes) 
        self.__ploteoGraficos(p_acp,1)
        self.__ploteoGraficos(p_acp,2)
        self.__ploteoGraficos(p_acp,3)
        
    def __ploteoGraficos(self,p_acp, tipo):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8,4), dpi = 200)
        if tipo==1:
            p_acp.plot_plano_principal()
        elif tipo==2:
            p_acp.plot_circulo()
        elif tipo==3:
            p_acp.plot_sobreposicion()     
            plt.show()
        
    def HAC(self, m):
        tsne = TSNE(n_components=2, random_state=42)
        embedding = tsne.fit_transform(self.__df)

        silhouette_scores = []

        for n_clusters in range(2, 11):
            hac_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=m, affinity='euclidean')
            clusters = hac_model.fit_predict(embedding)

            silhouette_avg = silhouette_score(embedding, clusters)
            silhouette_scores.append(silhouette_avg)

        optimal_n_clusters = np.argmax(silhouette_scores) + 2

        hac_model = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage=m, affinity='euclidean')
        clusters = hac_model.fit_predict(embedding)
        embedding_with_clusters = np.column_stack((embedding, clusters))
        
        print('Silhouette Score:', max(silhouette_scores))
        self.__plot_hac(embedding, m, silhouette_scores, clusters, embedding_with_clusters)
        self.__agregar_modelo('TSNE', optimal_n_clusters, max(silhouette_scores))


    def __plot_hac(self, df, m, silhouette_scores, clusters, embedding_with_clusters):
        plt.figure(figsize=(10, 6))
        Z = linkage(df, method=m, metric='euclidean')
        dendrogram(Z)
        plt.title('Dendrograma del clustering jerárquico aglomerativo')
        plt.xlabel('Índices de la muestra')
        plt.ylabel('Distancia')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
        plt.title('Silhouette Score para diferentes números de clusters')
        plt.xlabel('Número de clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, 11))
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=embedding_with_clusters[:, 0], y=embedding_with_clusters[:, 1], hue=clusters, palette='tab10', legend='full')
        plt.title('Clustering de casas usando HAC')
        plt.xlabel('HAC Dimensión 1')
        plt.ylabel('HAC Dimensión 2')
        plt.legend(title='Cluster')
        
    def KMeans(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.__df)
        
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
    
        optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        print('Silhouette Score:', max(silhouette_scores))
        self.__plot_kmeans(scaled_data, cluster_labels, silhouette_scores)
        self.__agregar_modelo('TSNE', optimal_n_clusters, max(silhouette_scores))
        
    def __plot_kmeans(self, df, cluster_labels, silhouette_scores):
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Puntuación de la Silueta')
        plt.title('Silhouette Score')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[:, 0], y=df[:, 1], hue=cluster_labels, palette='tab10', legend='full')
        plt.title('Clustering de casas usando Kmeans')
        plt.xlabel('Kmeans Dimensión 1')
        plt.ylabel('Kmeans Dimensión 2')
        plt.legend(title='Cluster')
        plt.show()
        
    def UMAP(self):
        umapp = umap.UMAP(n_components=2, random_state=42)
        df_umap = umapp.fit_transform(self.__df)
        
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(df_umap)
            silhouette_avg = silhouette_score(df_umap, cluster_labels)
            silhouette_scores.append(silhouette_avg)
    
        optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(df_umap)

        print('Silhouette Score:', max(silhouette_scores))
        self.__plotUMAP(df_umap, cluster_labels, silhouette_scores)
        self.__agregar_modelo('TSNE', optimal_n_clusters, max(silhouette_scores))

    def __plotUMAP(self, df, cluster_labels, silhouette_scores):
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Puntuación de la Silueta')
        plt.title('Silhouette Score')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[:, 0], y=df[:, 1], hue=cluster_labels, palette='tab10', legend='full')
        plt.title('Clustering de casas usando UMAP')
        plt.xlabel('UMAP Dimensión 1')
        plt.ylabel('UMAP Dimensión 2')
        plt.legend(title='Cluster')
        plt.show()
    

    def TSNE(self):
        tsne = TSNE(n_components=2, random_state=42)
        df_tsne = tsne.fit_transform(self.__df)
        
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(df_tsne)
            silhouette_avg = silhouette_score(df_tsne, cluster_labels)
            silhouette_scores.append(silhouette_avg)
    
        optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(df_tsne)
        
        print('Silhouette Score:', max(silhouette_scores))
        self.__plotTSNE(df_tsne, cluster_labels, silhouette_scores)
        
        self.__agregar_modelo('TSNE', optimal_n_clusters, max(silhouette_scores))


    def __plotTSNE(self, df, cluster_labels, silhouette_scores):
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Puntuación de la Silueta')
        plt.title('Silhouette Score')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[:, 0], y=df[:, 1], hue=cluster_labels, palette='tab10', legend='full')
        plt.title('Clustering de casas usando TSNE')
        plt.xlabel('TSNE Dimensión 1')
        plt.ylabel('TSNE Dimensión 2')
        plt.legend(title='Cluster')
        plt.show()


class Supervisado(AnalisisDatosExploratorio):
    def __init__(self, df):
        self.__df = df 
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
    @property
    def x_train(self):
        return self.__x_train 
    @x_train.setter
    def x_train(self, p_x_train):
        self.__x_train = p_x_train
    @property
    def y_train(self):
        return self.__y_train 
    @y_train.setter
    def y_train(self, p_y_train):
        self.__y_train = p_y_train
    @property
    def x_test(self):
        return self.__x_test 
    @x_test.setter
    def x_test(self, p_x_test):
        self.__x_test = p_x_test     
    @property
    def y_test(self):
        return self.__y_test 
    @y_test.setter
    def y_test(self, p_y_test):
        self.__y_test = p_y_test


    def data_transform(self, ts, target):
        y = self.__df[target]
        
        numeric_columns = self.__df.select_dtypes(include='number').columns
        wotarget = [col for col in numeric_columns if col != target]
        
        x = self.__df[wotarget]
    
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
        
        numeric_columns_with_target = list(wotarget) + [target]
        self.__df = self.__df[numeric_columns_with_target]
    
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x_scaled, y, train_size=ts)

    def Knn(self, n_neighbors=5, labels=None):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.__x_train, self.__y_train) 
        self.__plot_knn(knn, n_neighbors, labels)
    
    def __plot_knn(self, knn, n_neighbors, labels=None):
        y_pred = knn.predict(self.__x_test.values)
    
        print("Las predicciones en Testing son: {}".format(y_pred))
        if labels is None:
            labels = []
        MC = confusion_matrix(self.__y_test, y_pred, labels=labels)
        
        indices = ModMC.indices_general(MC, labels)
        for k in indices:
            print("\n%s:\n%s"%(k,str(indices[k])))

    def DecisionTree(self, target, ss=None,md= None):
        if ss==None and md == None:
            decisiontree = DecisionTreeClassifier()
            decisiontree.fit(self.__x_train, self.__y_train)
        else:
            decisiontree = DecisionTreeClassifier(min_samples_split=ss, max_depth = md)
            decisiontree.fit(self.__x_train, self.__y_train)
        self.__plot_decisiontree(decisiontree, target)

    def __plot_decisiontree(self, decisiontree, target):
        fig, ax = plt.subplots(1,1, figsize=(40,35))
        plot_tree(decisiontree,
                  feature_names=list(self.__x_test.columns.values), 
                  filled=True,
                  fontsize=10,
                  impurity=False,
                  node_ids=True,
                  proportion=True,
                  rounded=True,
                  precision=2,
                  ax=ax)
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        plt.show()
        analisis_target= Analisis_Predictivo(self.__df,
                            predecir = target,
                            modelo   = decisiontree,
                            estandarizar = True,
                            train_size   = len(self.__x_train))
        resultados = analisis_target.fit_predict_resultados()
        print("Las predicciones en Testing son: {}".format(analisis_target.fit_predict()))
        importancia = np.array(analisis_target.modelo.feature_importances_)
        
        etiquetas   = np.array(analisis_target.predictoras)
        orden       = np.argsort(importancia)
        
        importancia = importancia[orden]
        etiquetas   = etiquetas[orden]
        
        print("Importancia de las variables: ",importancia, "\n")
        print(etiquetas)
        
        fig, ax = plt.subplots(1,1, figsize = (12,6), dpi = 200)
        ax.barh(etiquetas, importancia)
        plt.show()
   
    def Randomforest(self, target, n_estimators, criterion, min_samples_split):
        randomforest = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, min_samples_split = min_samples_split)
        randomforest.fit(self.__x_train, self.__y_train)
        self.__plot_randomforest(randomforest, target)
        
    def __plot_randomforest(self, randomforest, target):
        analisis_target= Analisis_Predictivo(self.__df,
                            predecir = target,
                            modelo   = randomforest,
                            estandarizar = True,
                            train_size   = len(self.__x_train))
        resultados = analisis_target.fit_predict_resultados()
        print("Las predicciones en Testing son: {}".format(analisis_target.fit_predict()))
        importancia = np.array(analisis_target.modelo.feature_importances_)
        
        etiquetas   = np.array(analisis_target.predictoras)
        orden       = np.argsort(importancia)
        
        importancia = importancia[orden]
        etiquetas   = etiquetas[orden]
        
        print("Importancia de las variables: ",importancia, "\n")
        print(etiquetas)
        
        fig, ax = plt.subplots(1,1, figsize = (12,6), dpi = 200)
        ax.barh(etiquetas, importancia)
        plt.show()
        
    def XGBoosting(self, target, predictoras, n_estimators, min_samples_split):
        xgboosting = GradientBoostingClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split)
        self.__plot_xgboosting(xgboosting, target, predictoras)
        
    def __plot_xgboosting(self, xgboosting, target, predictoras):
        analisis_target = Analisis_Predictivo(self.__df, 
                                               predecir = target, 
                                               modelo   = xgboosting,
                                               predictoras = [i for i in predictoras],
                                               train_size  = len(self.__x_train))
        resultados = analisis_target.fit_predict_resultados()

        importancia = np.array(analisis_target.modelo.feature_importances_)
        etiquetas   = np.array(analisis_target.predictoras)

        orden       = np.argsort(importancia)
        importancia = importancia[orden]
        etiquetas   = etiquetas[orden]
        
        print("Importancia de las variables: ",importancia, "\n")
        print(etiquetas)

        fig, ax = plt.subplots(1,1, figsize = (12,6), dpi = 200)
        ax.barh(etiquetas, importancia)
        plt.show()

    def ADABoosting(self, target, predictoras, min_samples_split, max_depth, criterion, n_estimators):
        instancia_tree = DecisionTreeClassifier(min_samples_split = min_samples_split, 
                                        max_depth = max_depth,
                                        criterion = criterion)
        adaboosting = AdaBoostClassifier(base_estimator = instancia_tree,
                                   n_estimators   = n_estimators)
        self.__plot_adaboosting(adaboosting, target, predictoras)
        
    def __plot_adaboosting(self, adaboosting, target, predictoras):
    
        
        analisis_target = Analisis_Predictivo(self.__df, 
                                               predecir = target, 
                                               modelo   = adaboosting,
                                               predictoras = [i for i in predictoras],
                                               train_size  = len(self.__x_train))
        resultados = analisis_target.fit_predict_resultados()

        importancia = np.array(analisis_target.modelo.feature_importances_)
        etiquetas   = np.array(analisis_target.predictoras)

        orden       = np.argsort(importancia)
        importancia = importancia[orden]
        etiquetas   = etiquetas[orden]
        
        print("Importancia de las variables: ",importancia, "\n")
        print(etiquetas)

        fig, ax = plt.subplots(1,1, figsize = (12,6), dpi = 200)
        ax.barh(etiquetas, importancia)
        plt.show()
        
    def benchmark(self):
        resultados = []

        kmeans_result = ('KMeans', self.KMeans())
        resultados.append(kmeans_result)

        umap_result = ('UMAP', self.UMAP())
        resultados.append(umap_result)

        tsne_result = ('TSNE', self.TSNE())
        resultados.append(tsne_result)

        hac_result = ('HAC', self.HAC())
        resultados.append(hac_result)

        self.__show_benchmarking(resultados)
    
    def __show_benchmarking(self, resultados):
        df_resultados = pd.DataFrame(resultados, columns=['Algoritmo', 'Número de Clusters', 'Silhouette Score'])
        print(df_resultados)


class Regresion(AnalisisDatosExploratorio):
    def __init__(self, df):
        self.__df = df 
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
    @property
    def x_train(self):
        return self.__x_train 
    @x_train.setter
    def x_train(self, p_x_train):
        self.__x_train = p_x_train
    @property
    def y_train(self):
        return self.__y_train 
    @y_train.setter
    def y_train(self, p_y_train):
        self.__y_train = p_y_train
    @property
    def x_test(self):
        return self.__x_test 
    @x_test.setter
    def x_test(self, p_x_test):
        self.__x_test = p_x_test     
    @property
    def y_test(self):
        return self.__y_test 
    @y_test.setter
    def y_test(self, p_y_test):
        self.__y_test = p_y_test


    def data_transform(self, ts, target):
        y = self.__df[target]
        
        numeric_columns = self.__df.select_dtypes(include='number').columns
        wotarget = [col for col in numeric_columns if col != target]
        
        x = self.__df[wotarget]
    
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
        
        numeric_columns_with_target = list(wotarget) + [target]
        self.__df = self.__df[numeric_columns_with_target]
    
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x_scaled, y, train_size=ts)

    def Linearsimple(self, predictora):
        x_train = self.__x_train[[predictora]]
        y_train = self.__y_train
        
        x_test = self.__x_test[[predictora]]
        y_test = self.__y_test
        
        model = LinearRegression()
        model.fit(x_train, y_train)
        
        r = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:", mse)
        print("MAE:", mae)
        print("R^2:", r)
        
    def Linearmultiple(self, predictoras):
        x_train = self.__x_train[predictoras]
        y_train = self.__y_train
        
        x_test = self.__x_test[predictoras]
        y_test = self.__y_test
        
        model = LinearRegression()
        model.fit(x_train, y_train)
        
        r = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:", mse)
        print("MAE:", mae)
        print("R^2:", r)
        
    def Ridge(self, alphas):
        x_train = self.__x_train
        y_train = self.__y_train
        
        x_test = self.__x_test
        y_test = self.__y_test
        
        modelo = RidgeCV(alphas = alphas, fit_intercept= True, store_cv_values = True)
        modelo.fit(X=x_train, y=y_train)
        
        ridge_model = Ridge(alpha=modelo.alpha_)
        ridge_model.fit(x_train, y_train)
        
        r = ridge_model.score(x_test, y_test)
        y_pred = ridge_model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:", mse)
        print("MAE:", mae)
        print("R^2:", r)
        
    def Lasso(self, alphas):
        x_train = self.__x_train
        y_train = self.__y_train
        
        x_test = self.__x_test
        y_test = self.__y_test
        
        modelo = LassoCV(alphas = alphas, fit_intercept= True, cv = 10)
        modelo.fit(X=x_train, y=y_train)
        
        lasso_model = Lasso(alpha=modelo.alpha_)
        lasso_model.fit(x_train, y_train)
        
        r = lasso_model.score(x_test, y_test)
        y_pred = lasso_model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:", mse)
        print("MAE:", mae)
        print("R^2:", r)
        
    def Forestregressor(self,  n_estimators=50, max_depth=2, min_samples_split=2, min_samples_leaf=1):
        x_train = self.__x_train
        y_train = self.__y_train
        
        x_test = self.__x_test
        y_test = self.__y_test
        
        modelo_Bosque = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=0)
        modelo_Bosque.fit(X=x_train, y=y_train)

        r = modelo_Bosque.score(x_test, y_test)
        y_pred = modelo_Bosque.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:", mse)
        print("MAE:", mae)
        print("R^2:", r)
        

    
            
            
            
            
            
            
            