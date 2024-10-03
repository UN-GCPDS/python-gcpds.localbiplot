# example.py

"""
============
local biplot
============
"""


# def greet(name):
#     """
#     This function greets the person whose name is passed as a parameter
#     """
#     return f"Hello, {name}!"

# if __name__ == "__main__":
#     print(greet("World"))

# libraries_to_install = {'adjustText': 'adjustText'} #'adjustText': 'adjustText',  'umap-learn': 'umap'

# for library, alias in libraries_to_install.items():
#     try:
#         exec(f"import {alias}")
#     except ImportError:
#         pip install {library} --quiet

 

import os
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import kdeplot
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize



class LocalBiplot():
  """
  A class to perform local biplot analysis using various dimensionality reduction techniques
  and affine transformations.

  Attributes:
  -----------
  redm : str
      The dimensionality reduction method to use ('umap' or 'tsne').
  affine_ : str
      Type of affine transformation ('full' or 'rotation').
  perplexity : str or int
      Perplexity parameter for t-SNE.
  min_dist : float
      Minimum distance parameter for UMAP.

  Methods:
  --------
  dim_red(X):
      Performs dimensionality reduction on the input data X, returning the transformed data.

  biplot2D(X, plot_=True, labels=None, loading_labels=None):
      Creates a 2D PCA biplot of the input data X, optionally displaying it with specified labels.

  local_biplot2D(X, y, plot_=True, loading_labels=None):
      Performs local biplot analysis on the input data X with labels y, visualizing the results if plot_ is True.

  affine_transformation(params, points):
      Applies an affine transformation to the input points using the given parameters.

  objective_function(params, source_points, target_points):
      Objective function to minimize the mean squared error between transformed source points and target points.

  affine_transformation_obj(source_points, target_points, initial_guess=np.array([1, 1, 0, 0, 0, 0, 0])):
      Optimizes the affine transformation parameters to match source points to target points using an initial guess.

  plot_arrows(means_, points, head_width=0.025, color='b', linestyle='-'):
      Plots arrows from means to points for visual representation in biplots.

  biplot_global(score, loading, rel_, axbiplot, axrel, mean_=None, labels=None, loading_labels=None, score_labels=None, bar_c='b'):
      Creates a global biplot for the first two principal components, incorporating additional visual elements like loading vectors.

  plot_colorbar(cmap, ax, label, vmin=None, vmax=None):
      Adds a colorbar to the given axis ax, labeled appropriately and scaled to specified minimum and maximum values.

  plot_correlation_matrix(corr_matrix, labels=None, title=None):
      Plots a heatmap of the correlation matrix with optional labels and title.

  nor_correlation(self, B):
      Computes the normalized correlation matrix for input matrix B (P x M) by calculating 
      the dot products of B and normalizing the results.
  """   
  
  def __init__(self,redm = 'umap',affine_='full',perplexity='auto',min_dist=0.75):
    """
      Initializes the LocalBiplot class.

      Args:
          redm (str): Dimensionality reduction method ('umap' or 'tsne').
          affine_ (str): Type of affine transformation ('full' or 'rotation').
          perplexity (float or str): Perplexity parameter for t-SNE; if 'auto', it is calculated based on the number of samples.
          min_dist (float): Minimum distance parameter for UMAP.
    """
    self.affine_ = affine_
    if affine_ == 'rotation': #si se  seleccion rotacion se asigna 90 grados y otros parametros
      self.bounds = ((1,1),(1,1),(0,0),(0,0),(-np.pi,np.pi),(0,0),(0,0))
    else:
       self.bounds = ((None,None),(None,None),(None,None),(None,None),(-np.pi,np.pi),(None,None),(None,None)) # solo 180 grados de rotacion (refleja)

    self.perplexity = perplexity
    self.min_dist = min_dist #parametro de distancia minimo
    self.redm = redm

  def dim_red(self,X):
    """
    Performs dimensionality reduction on the input data.

    Args:
        X (ndarray): Input data to be reduced.

    Returns:
        ndarray: Transformed data after dimensionality reduction.
    """
    if self.perplexity == 'auto':
      self.perplexity = np.round(0.5*np.sqrt(X.shape[0])) # se calcula el perpelxity segun la mitad de la raiz cuadrada del numero de muestras
    if self.redm == 'umap':
      self.red_ = UMAP(n_components=2,n_neighbors=int(self.perplexity),random_state=42, min_dist=self.min_dist)
    else:
      self.red_ = TSNE(n_components=2,perplexity=self.perplexity,random_state=42, init='pca')
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.red_.fit_transform(MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)))

  def biplot2D(self,X,plot_=True,labels=None,loading_labels=None, all=False, filename=None, nval=None): #biplot clasico
     """
        Creates a 2D biplot of the transformed data.

        Args:
            X (ndarray): Input data for biplot.
            plot_ (bool): Flag to indicate whether to plot the biplot.
            labels (list): Labels for data points in the plot.
            loading_labels (list): Labels for loading vectors.
            all (bool): Flag to return all data or only the first two components.
            filename (str): Name of the file to save the figure.
            nval (int or None): Additional parameter for plotting.

        Returns:
          Tuple: Loading, relative importance, and scores of the biplot.
     """
 
     pca = PCA( random_state = 42) # la transformacion de los datos es entre -1 y 1
     score = MinMaxScaler(feature_range=(-1, 1)).fit_transform(pca.fit_transform(MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)))
     loading = pca.components_.T
     rel_ = softmax((abs(loading[:,:2].dot(np.diag(pca.explained_variance_[:2])))).sum(axis=1)) # se calcula la relevancia de cada variable usando una softmax
    
     if plot_:
        fig,ax = plt.subplots(1,1,figsize=(9, 7))
        self.biplot_global(score, loading, rel_,labels=labels, loading_labels=loading_labels,axbiplot=ax,axrel=ax, nval=nval) # se dibujan las relevancias y el scatter del biplot
        #ax[0].set_title('2D PCA Global Biplot')
        if filename is not None:
          self.save_fig(filename, fig=fig, tight_layout=True, fig_extension="pdf", resolution=300)
        plt.show()

     if all:
         return loading,rel_, score
     else:
         return loading[:,:2],rel_,score[:,:2]#loading[:,:2]



  def local_biplot2D(self,X,y,plot_=True,corrplot_=True,loading_labels=None, filename=None, nval= None):
    """
    Creates a local biplot for the provided data.

    Args:
        X (ndarray): Input data for local biplot.
        y (ndarray or int): Cluster labels or number of clusters.
        plot_ (bool): Flag to indicate whether to plot the biplot.
        corrplot_ (bool): Flag to indicate whether to create a correlation plot.
        loading_labels (list): Labels for loading vectors.
        filename (str): Name of the file to save the figure.
        nval (int or None): Additional parameter for plotting.

    Returns:
        ndarray: Cluster labels assigned to the input data.
    """
    font_size=25
    print('Dimensionality Reduction...')
    X_ = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)#minmaxscaler between -1 +1
    Z = self.dim_red(X_) #Nonlinear Dimensionality Reduction
    if type(y) == int: #no labels -> clustering
       print('Performing clustering...')
       self.y = KMeans(n_clusters=y,random_state=42).fit_predict(Z)
       print(f'{self.y.shape} - {np.unique(self.y)}')
    else:
      self.y = y

    C_ = len(np.unique(self.y))
    Zl = np.zeros(Z.shape)
    loading_ = np.zeros((C_,X.shape[1],2))
    loading_r = np.zeros((C_,X.shape[1],2))
    rel_ = np.zeros((C_,X.shape[1]))
    opt_params = np.zeros((C_,7)) #affine transformation parameters se tiene un arreglo de parametros para cada cluster

    if plot_:
      fig,ax = plt.subplots(1,1, figsize=(10,7))
      cmap_ = mpl.colormaps['jet'].resampled(C_)
      cmap_ = cmap_(range(C_))

    if corrplot_:
      fontsize = 8
      fig3, ax3  = plt.subplots(2, C_+1, figsize=(20, 8))
      plt.subplots_adjust(wspace= 0.05, hspace= 0.1)
    # Add colorbar for last heatmap
      cbar_ax = fig3.add_axes([0.93, 0.15, 0.015, 0.68])
      cbar_ax.tick_params(labelsize=font_size)
      norm = Normalize(vmin=0, vmax=1)  # Customize normalization if needed
      sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
      sm.set_array([])
      fig3.colorbar(sm, cax=cbar_ax)
   

    print('Affine Transformation...')
    for c in np.unique(self.y):
      print(f'{c+1}/{C_}')
      loading_[c],rel_[c],Zl[self.y==c] = self.biplot2D(X_[self.y==c],plot_=False) #pca biplot on c-th group
      Zl[self.y==c], opt_params[c],_ = self.affine_transformation_obj(Zl[self.y==c],Z[self.y==c]) #affine transformation training on c-th group
      loading_r[c] = self.affine_transformation(opt_params[c],loading_[c]) #transform loadings on c-th group
   

     
      if corrplot_:

          # Plot correlation of input data
            sns.heatmap(np.abs(np.corrcoef(X_.T).round(3)),  ax=ax3[0, 0], vmin=0, vmax=1, cmap='Reds', cbar=False,
                        linecolor="w", linewidths=1, xticklabels=loading_labels, yticklabels=loading_labels)
            ax3[0, 0].set_ylabel('Input data', fontsize=font_size)
           # ax3[0, 0].set_xticks([])rotation=45
            ax3[0, 0].yaxis.set_ticklabels(loading_labels , rotation=90)
            ax3[1, 1].set_ylabel('Local Biplot', fontsize=font_size)
            ax3[0, 0].tick_params(axis='y', labelsize=font_size)
            ax3[1, 1].tick_params(axis='y', labelsize=font_size)
            ax3[0, 0].tick_params(axis='x', labelsize=font_size)
            ax3[1, 0].axis("off")
            if c == 0:
              state = loading_labels
            else:
              state = False

            # Plot correlation biplot matrix B
            sns.heatmap(np.abs(self.nor_correlation(loading_r[c])),  ax=ax3[1, c + 1],  # np.abs(loading_[c].dot(loading_[c].T))
                        vmin=0, vmax=1, robust=True, cmap='Reds', cbar=False, linecolor="w", linewidths=1,
                        xticklabels=loading_labels, yticklabels=state)
            ax3[1, c + 1].tick_params(axis='x', labelsize=font_size)
            #print(loading_r[c].dot( loading_r[c].T))
            # Plot correlation of input data matrix
            sns.heatmap(np.abs(np.corrcoef(X_[self.y==c].T).round(3)), vmin=0, vmax=1,  ax=ax3[0, c + 1], cmap='Reds',
                        cbar=False, linecolor="w", linewidths=1, yticklabels=False, xticklabels=False)
            # ax3[0, c + 1].yaxis.set_ticklabels([])
            ax3[0, c + 1].set_title('Cluster ' + str(c + 1), fontsize=font_size, color=cmap_[c])
            # ax3[1, c + 1].set_xticks([])rotation=45
            # #ax3[1, c + 1].yaxis.set_ticklabels([])

            # if c == 1:
            #
            #     ax3[1, 1].tick_params(axis='x', labelsize=font_size)
            #     ax3[1, 1].tick_params(axis='y', labelsize=fontsize)
            #     ax3[1, 1].xaxis.set_ticks(np.arange(len(loading_labels))+0.5)
            #     ax3[1, 1].xaxis.set_ticklabels(loading_labels, rotation=90)
            #     ax3[1, 1].yaxis.set_ticks(np.arange(len(loading_labels))+0.5)
            #     ax3[1, 1].yaxis.set_ticklabels(loading_labels)



      if plot_:

        mean_ = np.repeat(Z[self.y==c].mean(axis=0).reshape(1,-1),(self.y==c).sum(),axis=0)
        print(f'plot {c+1}-th group')

        self.biplot_global(Z[self.y==c], loading_r[c], rel_[c],labels=cmap_[c],mean_ = mean_, loading_labels= loading_labels, axbiplot=ax, axrel=ax,bar_c=cmap_[c], nval=nval)
        if filename is not None:
          self.save_fig(filename, fig=fig, tight_layout=True, fig_extension="pdf", resolution=300)

    # ax[0].set_xlabel('Emb. 1', fontsize=font_size)
    # ax[0].set_ylabel('Emb. 2')
    #ax[0].set_title(f'2D Local Biplot ({self.redm})')
    if corrplot_ and filename is not None:
      fig3.savefig(("correlation_"+ filename+".pdf"), format="pdf", dpi=300)
    plt.show()
    self.loadings_l = loading_r
    self.Zr = Z
    self.rel_l = rel_


    return self.y

  def save_fig(self, fig_name, fig, tight_layout=True, fig_extension="pdf", resolution=300):
    path = os.path.join(fig_name + "." + fig_extension)
    #print("Saving figure", fig_name)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, format=fig_extension, dpi=resolution)


  def affine_transformation(self,params,points):
    #points \in N x2
    #sx,sy,hx,hy,theta,tx,ty = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    S = np.array([[params[0],0],[0,params[1]]])
    H = np.array([[params[2],1],[1,params[3]]])
    R = np.array([[np.cos(params[4]),-np.sin(params[4])],[np.sin(params[4]),np.cos(params[4])]])
    M = R.dot(H).dot(S)
    tr_ = np.array([params[5],params[6]])
    return (M.dot(points.T)+np.repeat(tr_.reshape(-1,1), points.shape[0], axis=1)).T

  def objective_function(self, params, source_points, target_points):
      """
      The objective function to minimize: the mean squared error between the
      transformed source points and the target points.

      Parameters:
      - params: Parameters of the affine transformation.
      - source_points: Source points to transform. N x 2
      - target_points: Target points to match. N x 2

      Returns:
      - Mean squared error between transformed source points and target points.
      """
      transformed_points = self.affine_transformation(params, source_points)
      return np.mean(np.sum((transformed_points - target_points)**2, axis=1))


  def affine_transformation_obj(self, source_points,target_points,initial_guess = np.array([1, 1, 0, 0, 0, 0,0])):
      """
      Optimize the parameters for an affine transformation that matches the 
      source points to the target points.

      Parameters
      ----------
      source_points : numpy.ndarray
          An (N x 2) array of points that need to be transformed.
          
      target_points : numpy.ndarray
          An (N x 2) array of points to which the source points should be matched.
          
      initial_guess : numpy.ndarray, optional
          Initial guess for the parameters of the affine transformation. 
          Default is an identity matrix with zero translation, 
          represented as np.array([1, 1, 0, 0, 0, 0, 0]).

      Returns
      -------
      transformed_points : numpy.ndarray
          The points after applying the optimized affine transformation.
          
      optimized_params : numpy.ndarray
          The optimized parameters for the affine transformation.
          
      result : OptimizeResult
          The result of the optimization, including details about the optimization process.
      """
      #source_points, target_points N x 2
      # Initial guess for the parameters (identity matrix and zero translation)
      # Perform optimization
      result = minimize(self.objective_function, x0=initial_guess, bounds=self.bounds, args=(source_points, target_points))

      # Extract the optimized transformation matrix and translation vector
      optimized_params = result.x
      transformed_points = self.affine_transformation(optimized_params,source_points)
      return transformed_points, optimized_params, result

  def plot_arrows(self,means_,points,head_width=0.025,color='b',linestyle ='-'):
      """
      Plot arrows from specified means to points on a 2D plane.

      Parameters
      ----------
      means_ : numpy.ndarray
          An (N x 2) array of mean points from which arrows originate.
          
      points : numpy.ndarray
          An (N x 2) array of destination points where arrows point to.
          
      head_width : float, optional
          Width of the arrow heads. Default is 0.025.
          
      color : str, optional
          Color of the arrows. Default is 'b' (blue).
          
      linestyle : str, optional
          Style of the lines used for arrows. Default is '-' (solid line).

      Returns
      -------
      None
      """
      N,P = points.shape

      for n in range(N):
        plt.arrow(means_[n,0],means_[n,1],points[n,0],points[n,1],head_width=head_width,color=color,linestyle=linestyle)
      return
  def kdeplot_(self, data, hue_attr='Date', c_attr =''):
      """
      Create a KDE plot with scatter overlay for the specified data.

      Parameters
      ----------
      data : pandas.DataFrame
          The input data containing the variables for the KDE plot.
          
      hue_attr : str, optional
          Column name in the data for color encoding. Default is 'Date'.
          
      c_attr : str, optional
          Column name in the data for color scaling in the scatter plot. Default is an empty string.

      Returns
      -------
      None
      """
      fig, ax = plt.subplots(1,1,figsize=(10,5))
      dataZ = pd.DataFrame(self.Zr, columns = ['PC1', 'PC2'])
      #treatment
      kdeplot(data=dataZ, x='PC1', y='PC2',hue=data[hue_attr], ax=ax, levels=3)
      dataZ.plot(kind='scatter',x='PC1',y='PC2',ax=ax,c=data[c_attr],cmap='jet',legend=False)


  def biplot_global(self,score, loading, rel_,axbiplot,axrel,mean_ = None,labels=None, loading_labels=None, score_labels=None,bar_c='b', filename=None, nval=None):
    """
    Creates a biplot for the first two principal components.

    Parameters:
    - score: 2D array of PCA scores, typically pca.transform(X)
    - loading: 2D array of PCA loadings, typically pca.components_.T
    - loading_labels: list of strings, feature names. (optional)
    - score_labels: list of strings, sample labels. (optional)
    """

    xs = score[:, 0]
    ys = score[:, 1]
    n = loading.shape[0]
    font_size = 25
    if nval is not None:

      loading_order = np.argsort(rel_)[::-1]
      # Extraer las 3 primeras para biplot:
      plot_index =  loading_order[:nval]
      # # Extraer nombres de las variables a mostrar:
      names = loading_labels
      loading = loading[plot_index]
      plot_names = [names[i] for i in plot_index]
    else:
      nval = loading.shape[0]
      plot_names = loading_labels


  
    if mean_ is None:
      mean_ = np.zeros((n,2))

    # Plot scores
    if labels is not None:
      cmap_ = mpl.colormaps['jet'].resampled(10)
      #cmap_ = mpl.colormaps['jet'].resampled(len(np.unique(labels)))
      axbiplot.scatter(xs, ys, alpha=0.4,c=labels, cmap=cmap_, marker='.')
    else:
      axbiplot.scatter(xs, ys, alpha=0.4,cmap=cmap_, marker='.')

    if score_labels is not None:
        for i, txt in enumerate(score_labels):
            axbiplot.annotate(txt, (xs[i], ys[i]), fontsize=8)

    # Plot loading vectors
    for i in range(nval):

        axbiplot.arrow(mean_[i,0], mean_[i,1], loading[i, 0]*max(abs(xs)*1.2), loading[i, 1]*max(abs(ys)*1.2),
                       color='gray', alpha=0.5, head_width=0.025, head_length=0.025)
        if loading_labels is not None:
            axbiplot.text(mean_[i,0]+loading[i, 0]*max(abs(xs))*1.4, mean_[i,1]+loading[i, 1]*max(abs(ys))*1.4,
                          plot_names[i], color='k', ha='center', va='center', fontsize=12)

    axbiplot.set_xlabel("PC1",  fontsize=font_size)
    axbiplot.set_ylabel("PC2", fontsize=font_size)
    axbiplot.set_xticklabels([])
    axbiplot.set_yticklabels([])




    # axrel.bar(np.arange(1,n+1),rel_,color=bar_c, alpha=0.5)
    # axrel.set_xticks(np.arange(1,n+1),loading_labels,rotation=90)
    # axrel.set_ylabel("Normalized Relevance")
    # plt.show()

    # # Save the figure if filename is provided
    # if filename is not None:
    #    plt.savefig(filename,  bbox_inches='tight')

  def nor_correlation(self,B): #B \in P x M
    Corr_ = (B.T).dot(B)
    C_ = B.dot(B.T)
    dia = np.diag(C_).reshape(-1,1)
    nor_ = np.sqrt(dia.dot(dia.T))+1e-10
    return np.divide(C_,nor_) #P x P \in [0,1]

  def correlations_by_target(self,data,X,y, col_names = ['FA', 'Lp2n', 'Lp4n', 'ALL V'], filename=None,loading_labels=None, key='Variety'):


    fontsize = 13
    X_ = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)#minmaxscaler between -1 +1
    Z = self.dim_red(X_) #Nonlinear Dimensionality Reduction
    if type(y) == int: #no labels -> clustering
       print('Performing clustering...')
       self.y = KMeans(n_clusters=y,random_state=42).fit_predict(Z)
       print(f'{self.y.shape} - {np.unique(self.y)}')
    else:
      self.y = y


    indices_FA = data[data[key] == 1].index.tolist()
    indices_Lp2n = data[ data[key] == 2].index.tolist()
    indices_Lp4n = data[data[key] == 3].index.tolist()

    CORR_ALL_FA = np.abs(np.corrcoef(X_[indices_FA].T)).round(3)
    CORR_ALL_Lp2n = np.abs(np.corrcoef(X_[indices_Lp2n].T)).round(3)
    CORR_ALL_Lp4n = np.abs(np.corrcoef(X_[indices_Lp4n].T)).round(3)
    CORR_X_ = np.abs(np.corrcoef(X_.T)).round(3)

    ALL_corr_X_ = np.column_stack((CORR_ALL_FA[:-1, -1], CORR_ALL_Lp2n[:-1, -1], CORR_ALL_Lp4n[:-1, -1], CORR_X_[:-1, -1]))

    C_ = len(np.unique(self.y))
    cmap_ = mpl.colormaps['jet'].resampled(C_)
    cmap_ = cmap_(range(C_))
    plt.subplots_adjust(wspace= 0.05, hspace= 0.1)
    fig1, ax1 = plt.subplots(1, C_+1, figsize=(10, 10))
    fig2, ax2 = plt.subplots(1, C_+1, figsize=(10, 10))
    plt.subplots_adjust(wspace= 0.05, hspace= 0.1)
    Zl = np.zeros(Z.shape)
    loading_ = np.zeros((C_,X.shape[1],2))
    # loading_FA = np.zeros((C_,X.shape[1],2))
    # loading_Lp2n = np.zeros((C_,X.shape[1],2))
    # loading_Lp4n = np.zeros((C_,X.shape[1],2))
    loading_r = np.zeros((C_,X.shape[1],2))
    loading_r_FA = np.zeros((C_,X.shape[1],2))
    loading_r_Lp2n = np.zeros((C_,X.shape[1],2))
    loading_r_Lp4n = np.zeros((C_,X.shape[1],2))
    rel_ = np.zeros((C_,X.shape[1]))
    rel_FA = np.zeros((C_,X.shape[1]))
    rel_Lp2n = np.zeros((C_,X.shape[1]))
    rel_Lp4n = np.zeros((C_,X.shape[1]))
    opt_params = np.zeros((C_,7)) #affine transformation parameters se tiene un arreglo de parametros para cada cluster

    print('Affine Transformation...')
    for c in np.unique(self.y):
      print(f'{c+1}/{C_}')

      loading_[c],rel_[c],Zl[self.y==c] = self.biplot2D(X_[self.y==c],plot_=False) #pca biplot on c-th group
      Zl[self.y==c], opt_params[c],_ = self.affine_transformation_obj(Zl[self.y==c],Z[self.y==c]) #affine transformation training on c-th group
      loading_r[c] = self.affine_transformation(opt_params[c],loading_[c]) #transform loadings on c-th group




      sel_data = data.iloc[self.y==c]



      indices_FA = sel_data[sel_data[key] == 1].index.tolist()

      indices_Lp2n = sel_data[sel_data[key] == 2].index.tolist()
      indices_Lp4n = sel_data[sel_data[key] == 3].index.tolist()


      loading_[c],rel_[c],Zl[indices_FA] = self.biplot2D(X_[indices_FA],plot_=False)
      Zl[indices_FA], opt_params[c],_ = self.affine_transformation_obj(Zl[indices_FA],Z[indices_FA]) #affine transformation training on c-th group
      loading_r_FA[c] = self.affine_transformation(opt_params[c],loading_[c]) #transform loadings on c-th group


      loading_[c],rel_[c],Zl[indices_Lp2n] = self.biplot2D(X_[indices_Lp2n],plot_=False)
      Zl[indices_Lp2n], opt_params[c],_ = self.affine_transformation_obj(Zl[indices_Lp2n],Z[indices_Lp2n]) #affine transformation training on c-th group
      loading_r_Lp2n[c] = self.affine_transformation(opt_params[c],loading_[c]) #transform loadings on c-th group



      loading_[c],rel_[c], Zl[indices_Lp4n] = self.biplot2D(X_[indices_Lp4n],plot_=False)
      Zl[indices_Lp4n], opt_params[c],_ = self.affine_transformation_obj(Zl[indices_Lp4n],Z[indices_Lp4n]) #affine transformation training on c-th group
      loading_r_Lp4n[c] = self.affine_transformation(opt_params[c],loading_[c]) #transform loadings on c-th group


      corr_ALL_X_C = np.abs(np.corrcoef(X_[self.y==c].T).round(3))
      corr_FA_X_C =  np.abs(np.corrcoef(X_[indices_FA].T).round(3))
      corr_Lp2n_X_C = np.abs(np.corrcoef(X_[indices_Lp2n].T).round(3))
      corr_Lp4n_X_C =  np.abs(np.corrcoef(X_[indices_Lp4n].T).round(3))

      ALL_corr_X_C = np.column_stack((corr_FA_X_C[:-1, -1], corr_Lp2n_X_C[:-1, -1], corr_Lp4n_X_C[:-1, -1], corr_ALL_X_C[:-1, -1]))


      #Plot the non-linear local-Biplot SVD.
      #print(loading_[c].shape)
      corr_ALL = self.nor_correlation(loading_r[c]) #np.abs((loading_r[c].dot( loading_r[c].T))/(np.max([np.abs(loading_r[c].min()), np.abs(loading_r[c].max())])))
      corr_FA = self.nor_correlation(loading_r_FA[c]) #np.abs((loading_r_FA[c].dot( loading_r_FA[c].T))/(np.max([np.abs(loading_r_FA[c].min()), np.abs(loading_r_FA[c].max())])))
      corr_Lp2n = self.nor_correlation(loading_r_Lp2n[c]) # np.abs((loading_r_Lp2n[c].dot( loading_r_Lp2n[c].T))/(np.max([np.abs(loading_r_Lp2n[c].min()), np.abs(loading_r_Lp2n[c].max())])))
      corr_Lp4n = self.nor_correlation(loading_r_Lp4n[c])  #np.abs((loading_r_Lp4n[c].dot( loading_r_Lp4n[c].T))/(np.max([np.abs(loading_r_Lp4n[c].min()), np.abs(loading_r_Lp4n[c].max())])))

      ALL_corr = np.column_stack((corr_FA[:-1, -1], corr_Lp2n[:-1, -1], corr_Lp4n[:-1, -1], corr_ALL[:-1, -1]))

      # im = ax1[i].imshow(ALL_corr, vmin = 0, vmax=1, cmap='Reds')
      im = sns.heatmap(ALL_corr, vmin=0, vmax=1, cmap="Reds",cbar=False,ax=ax2[c+1], linewidths=.5,)
      ax2[c+1].set_title('Local Biplot\nCluster '+ str(c+1), color= cmap_[c], fontdict={'fontsize':10}, pad=12) # sns.heatmap(flights,cmap="YlGnBu"
      ax2[c+1].tick_params(axis='y', labelsize=fontsize-3)
      ax2[c+1].tick_params(axis='x', labelsize=fontsize-3)
      ax2[c+1].set_yticks([])
      ax2[c+1].xaxis.set_ticks(np.arange(len(col_names ))+0.5)
      ax2[c+1].xaxis.set_ticklabels(col_names , rotation=45)


      im_ = sns.heatmap(ALL_corr_X_C, vmin=0, vmax=1, cmap="Reds",cbar=False,ax=ax1[c+1], linewidths=.5,)
      ax1[c+1].set_title('Cluster '+ str(c+1), color= cmap_[c], fontdict={'fontsize':10}, pad=12) # sns.heatmap(flights,cmap="YlGnBu"
      ax1[c+1].tick_params(axis='y', labelsize=fontsize-3)
      ax1[c+1].tick_params(axis='x', labelsize=fontsize-3)
      ax1[c+1].set_yticks([])
      ax1[c+1].xaxis.set_ticks(np.arange(len(col_names ))+0.5)
      ax1[c+1].xaxis.set_ticklabels(col_names , rotation=45)

    # Perform operations and plot in the first cell the correlation of all data
    im1 = sns.heatmap(ALL_corr_X_ , vmin = 0, vmax=1, cmap='Reds', cbar=False, ax=ax1[0], linewidths=.5,)
    ax1[0].set_title('Input data',  fontdict={'fontsize':fontsize}, pad=12) # sns.heatmap(flights,cmap="YlGnBu"
    ax1[0].tick_params(axis='y', labelsize=fontsize-3)
    ax1[0].tick_params(axis='x', labelsize=fontsize-3)
    ax1[0].xaxis.set_ticks(np.arange(len(col_names ))+0.5)
    ax1[0].xaxis.set_ticklabels(col_names , rotation=45) #, rotation=90
    ax1[0].yaxis.set_ticks(np.arange(len(loading_labels[:-1]))+0.5)
    ax1[0].yaxis.set_ticklabels(loading_labels[:-1], rotation=0) #, rotation=90

    im2 = sns.heatmap(ALL_corr_X_ , vmin = 0, vmax=1, cmap='Reds', cbar=False, ax=ax2[0], linewidths=.5,)
    ax2[0].set_title('Input data',  fontdict={'fontsize':fontsize}, pad=12) # sns.heatmap(flights,cmap="YlGnBu"
    ax2[0].set_title('Input data',  fontdict={'fontsize':fontsize}, pad=12) # sns.heatmap(flights,cmap="YlGnBu"
    ax2[0].tick_params(axis='y', labelsize=fontsize-3)
    ax2[0].tick_params(axis='x', labelsize=fontsize-3)
    ax2[0].xaxis.set_ticks(np.arange(len(col_names ))+0.5)
    ax2[0].xaxis.set_ticklabels(col_names , rotation=45) #, rotation=90
    ax2[0].yaxis.set_ticks(np.arange(len(loading_labels[:-1]))+0.5)
    ax2[0].yaxis.set_ticklabels(loading_labels[:-1], rotation=0) #, rotation=90




  #cbar_ax = fig3.add_axes([0.93, 0.15, 0.015, 0.68])

    #Draw colorbar in the figure
    cbar_ax = fig2.add_axes([0.12, -0.02, 0.8, 0.03])
    cbar_ax1 = fig1.add_axes([0.12, -0.02, 0.8, 0.03])
    cbar_ax1.tick_params(labelsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize)
    norm = Normalize(vmin=0, vmax=1)  # Customize normalization if needed
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    fig2.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    fig2.tight_layout()
    fig1.colorbar(sm, cax=cbar_ax1, orientation='horizontal')
    fig1.tight_layout()
    if filename is not None:
      fig2.savefig('2just_target_correlations_'+ filename +'.pdf',  bbox_inches='tight')
      fig1.savefig('1just_target_correlations_'+ filename +'.pdf',  bbox_inches='tight')
    #fig1.savefig('local-biplot_SVD_grass.pdf', bbox_inches='tight')
    plt.show()



    # self.loadings_l = loading_r
    self.Zr = Z
    # self.rel_l = rel_


    return



    return





 


if __name__ == "__main__":
    rice_ = LocalBiplot(affine_='rotation',redm='umap') #class instance
    rice_.LocalBiplot_()
