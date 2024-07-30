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


#librerias a importar

import os
#import warnings
#warnings.filterwarnings("ignore")

import os
import warnings
warnings.filterwarnings("ignore")


import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl




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
      Performs dimensionality reduction on the input data X.
  biplot2D(X, plot_=True, labels=None, loading_labels=None):
      Creates a 2D PCA biplot of the input data X.
  local_biplot2D(X, y, plot_=True, loading_labels=None):
      Performs local biplot analysis on the input data X with labels y.
  affine_transformation(params, points):
      Applies an affine transformation to the input points using the given parameters.
  objective_function(params, source_points, target_points):
      Objective function to minimize the mean squared error between transformed source points and target points.
  affine_transformation_obj(source_points, target_points, initial_guess=np.array([1, 1, 0, 0, 0, 0, 0])):
      Optimizes the affine transformation parameters to match source points to target points.
  plot_arrows(means_, points, head_width=0.025, color='b', linestyle='-'):
      Plots arrows from means to points.
  biplot_global(score, loading, rel_, axbiplot, axrel, mean_=None, labels=None, loading_labels=None, score_labels=None, bar_c='b'):
      Creates a global biplot for the first two principal components.
  """
  def __init__(self,redm = 'umap',affine_='full',perplexity='auto',min_dist=0.75):
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
    Performs dimensionality reduction on the input data X using UMAP or t-SNE.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data.

    Returns:
    --------
    array-like, shape (n_samples, 2)
        The reduced dimensionality data.
    """
    if self.perplexity == 'auto':
      self.perplexity = np.round(0.5*np.sqrt(X.shape[0])) # se calcula el perpelxity segun la mitad de la raiz cuadrada del numero de muestras
    if self.redm == 'umap':
      self.red_ = UMAP(n_components=2,n_neighbors=int(self.perplexity),random_state=42, min_dist=self.min_dist)
    else:
      self.red_ = TSNE(n_components=2,perplexity=self.perplexity,random_state=42, init='pca')
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.red_.fit_transform(MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)))

  def biplot2D(self,X,plot_=True,labels=None,loading_labels=None, all=False, filename=None, nval=None): #biplot clasico
    # Example usage:
    # Assuming pca is your PCA object and X is the data you've fitted PCA on:
     pca = PCA( random_state = 42) # la transformacion de los datos es entre -1 y 1
     score = MinMaxScaler(feature_range=(-1, 1)).fit_transform(pca.fit_transform(MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)))
     loading = pca.components_.T
     rel_ = softmax((abs(loading[:,:2].dot(np.diag(pca.explained_variance_[:2])))).sum(axis=1)) # se calcula la relevancia de cada variable usando una softmax
     print("variance explained:", pca.explained_variance_ratio_)
     #print('pca_components\n', loading)
     #print('rel_', rel_)
     if plot_:
        fig,ax = plt.subplots(1,2,figsize=(20, 7))
        self.biplot_global(score, loading, rel_,labels=labels, loading_labels=loading_labels,axbiplot=ax[0],axrel=ax[1], nval=nval) # se dibujan las relevancias y el scatter del biplot
        ax[0].set_title('2D PCA Global Biplot')
        if filename is not None:
          self.save_fig(filename, fig=fig, tight_layout=True, fig_extension="pdf", resolution=300)
        plt.show()

     if all:
         return loading,rel_, score
     else:
         return loading[:,:2],rel_,score[:,:2]#loading[:,:2]



  def local_biplot2D(self,X,y,plot_=True,corrplot_=True,loading_labels=None, filename=None, nval= None):

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
      fig,ax = plt.subplots(1,2,figsize=(20, 7))
      cmap_ = mpl.colormaps['jet'].resampled(C_)
      cmap_ = cmap_(range(C_))

    if corrplot_:
      fontsize = 8
      fig3, ax3  = plt.subplots(2, C_+1, figsize=(20, 8), sharey ="all")
      plt.subplots_adjust(wspace= 0.05, hspace= 0.1)
    # Add colorbar for last heatmap
      cbar_ax = fig3.add_axes([0.93, 0.15, 0.015, 0.68])
      cbar_ax.tick_params(labelsize=fontsize)
      norm = Normalize(vmin=0, vmax=1)  # Customize normalization if needed
      sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
      sm.set_array([])
      fig3.colorbar(sm, cax=cbar_ax)
     # self.save_fig("correlation_"+ filename, fig=fig3, tight_layout=True, fig_extension="pdf", resolution=300)

    print('Affine Transformation...')
    for c in np.unique(self.y):
      print(f'{c+1}/{C_}')
      print(X_[self.y==c].shape)
      loading_[c],rel_[c],Zl[self.y==c] = self.biplot2D(X_[self.y==c],plot_=False) #pca biplot on c-th group
      Zl[self.y==c], opt_params[c],_ = self.affine_transformation_obj(Zl[self.y==c],Z[self.y==c]) #affine transformation training on c-th group
      loading_r[c] = self.affine_transformation(opt_params[c],loading_[c]) #transform loadings on c-th group
      #loading1_,rel1_,Zl1 = self.biplot2D(X_[self.y==c],plot_=False, all=True)
      #print(loading1_)

     # rel_[c] = softmax((abs(loading_r[c].dot(np.diag(explained_variance_)))).sum(axis=1)) # se calcula la relevancia de cada variable usando una softmax

      if corrplot_:

          # Plot correlation of input data
            sns.heatmap(np.abs(np.corrcoef(X_.T).round(3)),  ax=ax3[0, 0], vmin=0, vmax=1, cmap='Reds', cbar=False,
                        linecolor="w", linewidths=1, xticklabels=loading_labels, yticklabels=loading_labels)
            ax3[0, 0].set_ylabel('Input data', fontsize=12)
            ax3[0, 0].tick_params(axis='y', labelsize=fontsize)
            ax3[0, 0].tick_params(axis='x', labelsize=fontsize)
            ax3[1, 0].axis("off")

            # Plot correlation biplot matrix B
            sns.heatmap(np.abs(self.nor_correlation(loading_r[c])),  ax=ax3[1, c + 1],  # np.abs(loading_[c].dot(loading_[c].T))
                        vmin=0, vmax=1, robust=True, cmap='Reds', cbar=False, linecolor="w", linewidths=1,
                        yticklabels=loading_labels)
            #print(loading_r[c].dot( loading_r[c].T))
            # Plot correlation of input data matrix
            sns.heatmap(np.abs(np.corrcoef(X_[self.y==c].T).round(3)), vmin=0, vmax=1,  ax=ax3[0, c + 1], cmap='Reds',
                        cbar=False, linecolor="w", linewidths=1, xticklabels=False)
            ax3[0, c + 1].yaxis.set_ticklabels([])
            ax3[0, c + 1].set_title('Cluster ' + str(c + 1), fontsize=12, color=cmap_[c])
            ax3[1, c + 1].set_xticks([])
            #ax3[1, c + 1].yaxis.set_ticklabels([])

            if c == 1:
                ax3[1, c].set_ylabel('Local Biplot', fontsize=12)
                ax3[1, 1].tick_params(axis='x', labelsize=fontsize)
                ax3[1, 1].tick_params(axis='y', labelsize=fontsize)
                ax3[1, 1].xaxis.set_ticks(np.arange(len(loading_labels))+0.5)
                ax3[1, 1].xaxis.set_ticklabels(loading_labels, rotation=90)
                ax3[1, 1].yaxis.set_ticks(np.arange(len(loading_labels))+0.5)
                ax3[1, 1].yaxis.set_ticklabels(loading_labels)



      if plot_:

        mean_ = np.repeat(Z[self.y==c].mean(axis=0).reshape(1,-1),(self.y==c).sum(),axis=0)
        print(f'plot {c+1}-th group')

        self.biplot_global(Z[self.y==c], loading_r[c], rel_[c],labels=cmap_[c],mean_ = mean_, loading_labels= loading_labels, axbiplot=ax[0],axrel=ax[1],bar_c=cmap_[c], nval=nval)
        if filename is not None:
          self.save_fig(filename, fig=fig, tight_layout=True, fig_extension="pdf", resolution=300)

    ax[0].set_xlabel('Emb. 1')
    ax[0].set_ylabel('Emb. 2')
    ax[0].set_title(f'2D Local Biplot ({self.redm})')
    if filename is not None:
      fig3.savefig(("correlation_"+ filename+".pdf"), format="pdf", dpi=300)
    plt.show()
    self.loadings_l = loading_r
    self.Zr = Z
    self.rel_l = rel_


    return self.y

  def save_fig(self, fig_name, fig, tight_layout=True, fig_extension="pdf", resolution=300):
    path = os.path.join(fig_name + "." + fig_extension)
    print("Saving figure", fig_name)
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
      #source_points, target_points N x 2
      # Initial guess for the parameters (identity matrix and zero translation)
      # Perform optimization
      result = minimize(self.objective_function, x0=initial_guess, bounds=self.bounds, args=(source_points, target_points))

      # Extract the optimized transformation matrix and translation vector
      optimized_params = result.x
      transformed_points = self.affine_transformation(optimized_params,source_points)
      return transformed_points, optimized_params, result

  def plot_arrows(self,means_,points,head_width=0.025,color='b',linestyle ='-'):
      N,P = points.shape

      for n in range(N):
        plt.arrow(means_[n,0],means_[n,1],points[n,0],points[n,1],head_width=head_width,color=color,linestyle=linestyle)
      return

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
    if nval is not None:

      loading_order = np.argsort(rel_)[::-1]
      print('my_ranking',loading_order)
      #print('nval', nval)
      # Extraer las 3 primeras para biplot:
      plot_index =  loading_order[:nval]
      # # Extraer nombres de las variables a mostrar:
      names = loading_labels
      loading = loading[plot_index]
     # print(loading)
      plot_names = [names[i] for i in plot_index]
      #print(loading[plot_index, :])
    else:
      nval = loading.shape[0]
      plot_names = loading_labels


    print(plot_names)
    if mean_ is None:
      mean_ = np.zeros((n,2))

    # Plot scores
    if labels is not None:
      cmap_ = mpl.colormaps['jet'].resampled(10)
      #cmap_ = mpl.colormaps['jet'].resampled(len(np.unique(labels)))
      axbiplot.scatter(xs, ys, alpha=0.3,c=labels, cmap=cmap_)
    else:
      axbiplot.scatter(xs, ys, alpha=0.3,cmap=cmap_)

    if score_labels is not None:
        for i, txt in enumerate(score_labels):
            axbiplot.annotate(txt, (xs[i], ys[i]), fontsize=8)

    # Plot loading vectors
    for i in range(nval):

        axbiplot.arrow(mean_[i,0], mean_[i,1], loading[i, 0]*max(abs(xs)*1.5), loading[i, 1]*max(abs(ys)*1.5),
                       color='gray', alpha=0.5, head_width=0.025, head_length=0.025)
        if loading_labels is not None:
            axbiplot.text(mean_[i,0]+loading[i, 0]*max(abs(xs))*1.75, mean_[i,1]+loading[i, 1]*max(abs(ys))*1.75,
                          plot_names[i], color='k', ha='center', va='center')

    axbiplot.set_xlabel("PC1")
    axbiplot.set_ylabel("PC2")
    axbiplot.set_xticklabels([])
    axbiplot.set_yticklabels([])


    axrel.bar(np.arange(1,n+1),rel_,color=bar_c, alpha=0.5)
    axrel.set_xticks(np.arange(1,n+1),loading_labels,rotation=90)
    axrel.set_ylabel("Normalized Relevance")

    #plt.show()

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



      # print("indices_FA", indices_FA)
      # print("indices_Lp2n", indices_Lp2n)
      # print("indices_Lp4n", indices_Lp4n)
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
      #print(corr_ALL.shape)
      corr_FA = self.nor_correlation(loading_r_FA[c]) #np.abs((loading_r_FA[c].dot( loading_r_FA[c].T))/(np.max([np.abs(loading_r_FA[c].min()), np.abs(loading_r_FA[c].max())])))
      corr_Lp2n = self.nor_correlation(loading_r_Lp2n[c]) # np.abs((loading_r_Lp2n[c].dot( loading_r_Lp2n[c].T))/(np.max([np.abs(loading_r_Lp2n[c].min()), np.abs(loading_r_Lp2n[c].max())])))
      corr_Lp4n = self.nor_correlation(loading_r_Lp4n[c])  #np.abs((loading_r_Lp4n[c].dot( loading_r_Lp4n[c].T))/(np.max([np.abs(loading_r_Lp4n[c].min()), np.abs(loading_r_Lp4n[c].max())])))

      ALL_corr = np.column_stack((corr_FA[:-1, -1], corr_Lp2n[:-1, -1], corr_Lp4n[:-1, -1], corr_ALL[:-1, -1]))

      # im = ax1[i].imshow(ALL_corr, vmin = 0, vmax=1, cmap='Reds')
      im = sns.heatmap(ALL_corr, vmin=0, vmax=1, cmap="Reds",cbar=False,ax=ax2[c+1], linewidths=.5,)
      ax2[c+1].set_title('Cluster '+ str(c+1), color= cmap_[c], fontdict={'fontsize':10}, pad=12) # sns.heatmap(flights,cmap="YlGnBu"
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

    # #Draw colorbar in the figure
    # cbar_ax = fig2.add_axes([0.12, -0.02, 0.8, 0.03])
    # cbar_ax1 = fig1.add_axes([0.12, -0.02, 0.8, 0.03])

    # Adjust position and size of the colorbars on the left side
    # Adjust position and size of the colorbars on the right side
    cbar_ax = fig2.add_axes([1.01, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar_ax1 = fig1.add_axes([1.01, 0.15, 0.03, 0.7])  # [left, bo
    cbar_ax1.tick_params(labelsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize)
    norm = Normalize(vmin=0, vmax=1)  # Customize normalization if needed
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    fig2.colorbar(sm, cax=cbar_ax, orientation='vertical')
    fig2.tight_layout()
    fig1.colorbar(sm, cax=cbar_ax1, orientation='vertical')
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
