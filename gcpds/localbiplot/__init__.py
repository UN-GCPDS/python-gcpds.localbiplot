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
    if affine_ == 'rotation':
      self.bounds = ((1,1),(1,1),(0,0),(0,0),(-np.pi,np.pi),(0,0),(0,0))
    else:
       self.bounds = ((None,None),(None,None),(None,None),(None,None),(-np.pi,np.pi),(None,None),(None,None))

    self.perplexity = perplexity
    self.min_dist = min_dist
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
      self.perplexity = np.round(0.5*np.sqrt(X.shape[0]))
    if self.redm == 'umap':
      self.red_ = UMAP(n_components=2,n_neighbors=int(self.perplexity),random_state=42, min_dist=self.min_dist)
    else:
      self.red_ = TSNE(n_components=2,perplexity=self.perplexity,random_state=42, init='pca')
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.red_.fit_transform(MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)))

  def biplot2D(self,X,plot_=True,labels=None,loading_labels=None):
      """
      Creates a 2D PCA biplot of the input data X.

      Parameters:
      -----------
      X : array-like, shape (n_samples, n_features)
          The input data.
      plot_ : bool, optional, default=True
          Whether to plot the biplot.
      labels : array-like, shape (n_samples,), optional
          Labels for the data points.
      loading_labels : list of str, optional
          Labels for the loadings.

      Returns:
      --------
      loading : array-like, shape (n_features, 2)
          The loadings for the first two principal components.
      rel_ : array-like, shape (n_features,)
          The relevance of each loading.
      score : array-like, shape (n_samples, 2)
          The PCA scores for the first two principal components.
      """
      # Example usage:
      # Assuming pca is your PCA object and X is the data you've fitted PCA on:
      pca = PCA(random_state = 42)
      score = MinMaxScaler(feature_range=(-1, 1)).fit_transform(pca.fit_transform(MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)))
      loading = pca.components_.T
      rel_ = softmax((abs(loading.dot(np.diag(pca.explained_variance_)))).sum(axis=1))

      if plot_:
        fig,ax = plt.subplots(1,2,figsize=(20, 7))
        self.biplot_global(score, loading, rel_,labels=labels, loading_labels=loading_labels,axbiplot=ax[0],axrel=ax[1])
        ax[0].set_title('2D PCA Global Biplot')
        plt.show()

      return loading[:,:2],rel_,score[:,:2]



  def local_biplot2D(self,X,y,plot_=True,loading_labels=None):
    """
    Performs local biplot analysis on the input data X with labels y.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,) or int
        The labels for the data points, or the number of clusters for k-means clustering.
    plot_ : bool, optional, default=True
        Whether to plot the biplot.
    loading_labels : list of str, optional
        Labels for the loadings.

    Returns:
    --------
    None
    """

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
    opt_params = np.zeros((C_,7)) #affine transformation parameters

    if plot_:
      fig,ax = plt.subplots(1,2,figsize=(20, 7))
      cmap_ = mpl.colormaps['jet'].resampled(C_)
      cmap_ = cmap_(range(C_))

    print('Affine Transformation...')
    for c in np.unique(self.y):
      print(f'{c+1}/{C_}')
      loading_[c],rel_[c],Zl[self.y==c] = self.biplot2D(X_[self.y==c],plot_=False) #pca biplot on c-th group
      Zl[self.y==c], opt_params[c],_ = self.affine_transformation_obj(Zl[self.y==c],Z[self.y==c]) #affine transformation training on c-th group
      loading_r[c] = self.affine_transformation(opt_params[c],loading_[c]) #transform loadings on c-th group

      if plot_:
        mean_ = np.repeat(Z[self.y==c].mean(axis=0).reshape(1,-1),(self.y==c).sum(),axis=0)
        print(f'plot {c+1}-th group')

        self.biplot_global(Z[self.y==c], loading_r[c], rel_[c],labels=cmap_[c],mean_ = mean_, loading_labels=loading_labels,axbiplot=ax[0],axrel=ax[1],bar_c=cmap_[c])
    ax[0].set_xlabel('Emb. 1')
    ax[0].set_ylabel('Emb. 2')
    ax[0].set_title(f'2D Local Biplot ({self.redm})')
    plt.show()
    self.loadings_l = loading_r
    self.Zr = Z
    self.rel_l = rel_
    return


  def affine_transformation(self,params,points):
    """
    Applies an affine transformation to the input points using the given parameters.

    Parameters:
    -----------
    params : array-like, shape (7,)
        The parameters for the affine transformation.
    points : array-like, shape (n_samples, 2)
        The points to transform.

    Returns:
    --------
    array-like, shape (n_samples, 2)
        The transformed points.
    """

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
      -----------
      - params: Parameters of the affine transformation.
      - source_points: Source points to transform. N x 2
      - target_points: Target points to match. N x 2

      Returns:
      --------
      - Mean squared error between transformed source points and target points.
      """

      transformed_points = self.affine_transformation(params, source_points)
      return np.mean(np.sum((transformed_points - target_points)**2, axis=1))


  def affine_transformation_obj(self, source_points,target_points,initial_guess = np.array([1, 1, 0, 0, 0, 0,0])):
      """
      Optimizes the affine transformation parameters to match source points to target points.

      Parameters:
      -----------
      source_points : array-like, shape (n_samples, 2)
          The source points to transform.
      target_points : array-like, shape (n_samples, 2)
          The target points to match.
      initial_guess : array-like, shape (7,), optional
          Initial guess for the affine transformation parameters.

      Returns:
      --------
      array-like, shape (n_samples, 2)
          The transformed source points.
      array-like, shape (7,)
          The optimized affine transformation parameters.
      scipy.optimize.OptimizeResult
          The result of the optimization.
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
      Plots arrows from means to points.

      Parameters:
      -----------
      means_ : array-like, shape (n_samples, 2)
          The starting points of the arrows.
      points : array-like, shape (n_samples, 2)
          The ending points of the arrows.
      head_width : float, optional
          The width of the arrow heads.
      color : str, optional
          The color of the arrows.
      linestyle : str, optional
          The line style of the arrows.

      Returns:
      --------
      None
      """
      N,P = points.shape

      for n in range(N):
        plt.arrow(means_[n,0],means_[n,1],points[n,0],points[n,1],head_width=head_width,color=color,linestyle=linestyle)
      return

  def biplot_global(self,score, loading, rel_,axbiplot,axrel,mean_ = None,labels=None, loading_labels=None, score_labels=None,bar_c='b'):
    """
    Creates a global biplot for the first two principal components.

    Parameters:
    -----------
    score : array-like, shape (n_samples, 2)
        The PCA scores for the first two principal components.
    loading : array-like, shape (n_features, 2)
        The loadings for the first two principal components.
    rel_ : array-like, shape (n_features,)
        The relevance of each loading.
    axbiplot : matplotlib.axes.Axes
        The axes for the biplot.
    axrel : matplotlib.axes.Axes
        The axes for the relevance plot.
    mean_ : array-like, shape (n_samples, 2), optional
        The mean values for the data points.
    labels : array-like, shape (n_samples,), optional
        The labels for the data points.
    loading_labels : list of str, optional
        The labels for the loadings.
    score_labels : list of str, optional
        The labels for the scores.
    bar_c : str, optional
        The color of the relevance bars.

    Returns:
    --------
    None
    """

    xs = score[:, 0]
    ys = score[:, 1]
    n = loading.shape[0]

    if mean_ is None:
      mean_ = np.zeros((n,2))

    # Plot scores
    if labels is not None:
      axbiplot.scatter(xs, ys, alpha=0.5,c=labels)
    else:
      axbiplot.scatter(xs, ys, alpha=0.5)

    if score_labels is not None:
        for i, txt in enumerate(score_labels):
            axbiplot.annotate(txt, (xs[i], ys[i]), fontsize=8)

    # Plot loading vectors
    for i in range(n):

        axbiplot.arrow(mean_[i,0], mean_[i,1], loading[i, 0]*max(abs(xs)), loading[i, 1]*max(abs(ys)),
                      color='r', alpha=0.5, head_width=0.025, head_length=0.05)
        if loading_labels is not None:
            axbiplot.text(mean_[i,0]+loading[i, 0]*max(abs(xs))*1.15, mean_[i,1]+loading[i, 1]*max(abs(ys))*1.15,
                     loading_labels[i], color='g', ha='center', va='center')

    axbiplot.set_xlabel("PC1")
    axbiplot.set_ylabel("PC2")


    axrel.bar(np.arange(1,n+1),rel_,color=bar_c)
    axrel.set_xticks(np.arange(1,n+1),loading_labels,rotation=90)
    axrel.set_ylabel("Normalized Relevance")
    #plt.show()

    return


if __name__ == "__main__":
    rice_ = LocalBiplot(affine_='rotation',redm='umap') #class instance
    rice_.LocalBiplot_()
