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


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import pandas as pd
import seaborn as sns
from seaborn import kdeplot

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

from scipy.spatial.distance import cdist, squareform
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy import interpolate

#from adjustText import adjust_text



# Objeto para guardar resultados:
class GMDOutput:
    pass

#centering kernels
class LocalBiplot(): #Poner en CamelCase

    """
    Object for data analysis using linear and non-linear Biplots obtained by
    SVD decomposition and a Generalized SVD decomposition .

    This class implements a set of functions for data analysis, including
    scaling, dimensionality reduction, kernel calculation, and biplots
    computation and display.

    Attributes
    ----------
    X:  pd.dataframe
        Input matrix of shape N x P.
    labels: array-like, optional
        Labels for the samples (default is None).
    perplexity: int or None, optional
        Perplexity for t-SNE (default is calculated as the square root of N).
    red: {'tsne', 'pca', 'umap'}, default is 'tsne'
        Dimensionality reduction method ('tsne' by default).
    sca: {'minmax'}, default is 'minmax'
        Data scaling method ('minmax' by default).
    random_seed: int, default is 123
        Seed for result reproducibility.

    Methods
    -------
    data_scaler(X, feature_range=(0, 1))
        Scale the data using MinMaxScaler if 'sca' is set to 'minmax'.
    reduce_dimensions(X)
        Reduce the dimensionality of the data using t-SNE, PCA, or UMAP.
    krbf(X)
        Calculate the Radial Basis Function (RBF) kernel matrix for the input data.
    center_kernel(K)
        Center a given kernel matrix using the Kernel Centering method.
    laplacian_score(X, K, tol=1e-10)
        Calculate the Laplacian score for a given dataset and kernel matrix.
    lnkbp_()
        Process and analyze the data through steps such as scaling, dimensionality reduction,
        kernel calculations, and Laplacian Score computation.
    localbp_(X_)
          Perform a local biplot operation on the scaled data (currently commented out).
    laplacian_score(X, K, tol=1e-10)
        Calculate the Laplacian score for a given dataset and kernel matrix
    GMD(X, H, Q, K)
        Generalized Matrix Decomposition method (power method) for a given dataset and kernel matrices.
    biplot_gmd_body(fit, index=None, names=None, sample_col='grey50', sample_pch=19, arrow_col='orange', arrow_cex=1)
        Generate a GMD-biplot based on generalized matrix decomposition results.
    plot_lnkbp_(hue, c, figsize=(25, 10))
        Plot various visualizations, including scatter plots, kernel matrices, and feature relevance.
    affine_transformM(parameters, array_A)
        Apply an affine transformation to the input array using the given parameters.
    registration_errorM(parameters, array_A, array_B)
        Compute the registration error between two sets of 2D points after applying an affine transformation.


    ...

    """
    def __init__(self, X, labels = None, perplexity = None, red = 'tsne', sca = 'minmax',random_seed=123):
          self.X = X.copy()
          self.columns_= self.X.columns
          self.labels = labels
          if perplexity==None:
            perplexity = round(np.sqrt(X.shape[0]))
          self.perplexity = perplexity
          self.random_seed = random_seed
          self.red = red
          self.sca = sca





    def data_scaler(self,X,feature_range=(0,1)):
        """
        this method scale the input data using MinMaxScaler if 'sca' is set to 'minmax'.

        Parameters
        ----------
        - X (array-like): Input matrix of shape N x P. Input data to be scaled.
        - feature_range (tuple, optional): Tuple specifying the minimum and maximum values of the feature range.
          Defaults to (0, 1).

        Returns
        ----------
        - An N x P scaled data matrix.
        """

        # Check if scaling method is 'minmax'
        if self.sca == 'minmax':
          # Create MinMaxScaler instance with specified feature_range
          sca_ = MinMaxScaler(feature_range=feature_range)
          # Fit and transform the data using the scaler
        return sca_.fit_transform(X)


    def reduce_dimensions(self, X):
        """
        Reduce the dimensionality of the input data using t-SNE, PCA, or UMAP.

        Parameters:
        -----------
        - X (array-like): Input matrix of shape N x P. Input data to be dimensionality reduced.

        Returns:
        --------
        - An n x 2 array-like dimensionality reduced data.
        """

        # Choose the dimensionality reduction method
        if self.red == 'tsne':
          self.reduce_dimensions = TSNE(n_components = 2, perplexity = self.perplexity, random_state=self.random_seed)
        elif self.red == 'pca':
          self.reduce_dimensions = PCA(n_components = 2, random_state=self.random_seed)
        # Uncomment the following lines if you use UMAP
        # elif self.red == 'umap' :
        #   self.reduce_dimension = UMAP(n_components = 2, n_neighbors = round(np.sqrt(X.shape[0])),min_dist =0.9,random_state=self.random_seed)
        else:
          # Use t-SNE as a default if the specified method is not recognized
          self.reduce_dimensions = TSNE(n_components = 2, perplexity = self.perplexity, random_state=self.random_seed)
        # Return the dimensionality reduced data after scaling
        return self.data_scaler(self.reduce_dimensions.fit_transform(X))





   
    def LocalBiplot_(self):
        """
        Process and analyze the data using a series of steps, including scaling, dimensionality reduction,
        kernel calculations, and Laplacian score computation.

        Returns:
        --------
        - YourClass instance: The modified instance with processed and analyzed data.
        """

        # Step 1: Scale the input data
        self.X_ = self.data_scaler(self.X) #scaler X
        # Step 2: Reduce dimensions using a chosen method
        self.Z = self.reduce_dimensions(self.X_) #red
        # Step 3: Perform a local biplot operation on the scaled data
        #self.localbp_(self.X_)
        # Step 4: Add the reduced dimensions to the original data
        self.X['P1'] = self.Z[:,0] #add red to pd
        self.X['P2'] = self.Z[:,1]
        # Step 5: Calculate kernel matrices for X scaled and Z
        
        return self

   



    def plot_transformed_clusters(self, ax, ZcA, VA, cmap,  arrow_size = 0.05 ):
        """
        Plot the non-linear local-Biplot SVD.

        Parameters:
        ----------
        - ax (matplotlib.axes._subplots.AxesSubplot): Axes on which to plot.
        - ZcA (numpy.ndarray): Transformed points of the cluster.
        - VA (numpy.ndarray): Transformed vector arrows of the cluster.
        - cmap: Color map for the scatter plot.
        - arrow_size

        Returns:
        --------
        None
        """

        texts = []
        # Calculate mean of transformed points
        ZcA_mean = ZcA.mean(axis=0)



        # Set arrow size and color



        # Scatter plot of transformed points
        ax.scatter(ZcA[:, 0], ZcA[:, 1], alpha=0.7, c=cmap)

        # Extract arrow coordinates from transformed vector arrows
        VA = VA /np.linalg.norm(VA,axis=0)

        arrow_x = VA[:, 0]
        arrow_y = VA[:, 1]

        # Calculate maximum values for scaling
        max_xlab = np.max(np.abs(ZcA[:, 0]))
        max_ylab = np.max(np.abs(ZcA[:, 1]))
        max_xarrow = np.max(np.abs(arrow_x))
        max_yarrow = np.max(np.abs(arrow_y))
        xratio = max_xarrow / max_xlab
        yratio = max_yarrow / max_ylab

        # Set axis points for arrow scaling
        xaxp = np.linspace(-max_xlab, max_xlab, num=5)
        yaxp = np.linspace(-max_ylab, max_ylab, num=5)

        xlab_arrow = xaxp * xratio
        ylab_arrow = yaxp * yratio

        # Calculate mean of transformed points for arrow origin
        ZcA_mean = ZcA.mean(axis=0)

        # Plot vector arrows and labels
        for k in range(VA.shape[0]):
            ax.arrow(ZcA_mean[0], ZcA_mean[1], (arrow_x[k] / xratio)*0.5, (arrow_y[k] / yratio)*0.5,
                     head_width= arrow_size, head_length= arrow_size, color='gray', linewidth=1.1)
            ax.text(((arrow_x[k]) / xratio)*0.5 + 0.03 + ZcA_mean[0], (arrow_y[k] / yratio)*0.5 + ZcA_mean[1]+ 0.01,
                     s='f' + str(k + 1), fontsize=16, color='black')

        
        #ax.set_xlim((-1,1))
        ax.set_ylim((-1,1.4))
        # Set plot title and axis labels
       # ax.set_title('Non-linear local-biplot SVD', fontsize=20)
        ax.set_xlabel('Dimension 1', fontsize=20)
        #ax.set_ylabel('PC 2', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        #ax.set_yticks([])
        # Remove the upper and right spines
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)

        # Display grid
        #ax.grid(True)



    def affine_transformM(self, parameters, array_A):
        """
          Apply an affine transformation to the input array using the given parameters.

          Parameters:
          ----------
          - parameters (array-like): Affine transformation parameters.
              - parameters[0]: Scaling factor
              - parameters[1]: Rotation angle (in radians)
              - parameters[2:]: Translation along x and y axes
          - array_A (array-like): Input array to be transformed.

          Returns:
          --------
          - array-like: Transformed array after applying the affine transformation.

          """
        # Apply affine transformation to array_A with parameters
        N = array_A.shape[0] #N x 2 array
        scale = parameters[0]
        rotation = parameters[1]
        translation = parameters[2:]
        transformation_matrix = np.array([
            [scale * np.cos(rotation), -scale * np.sin(rotation), translation[0]],
            [scale * np.sin(rotation), scale * np.cos(rotation), translation[1]],
        ]) # 2 x 3 transformation matrix
        transformed_A = np.dot(transformation_matrix, np.c_[array_A,np.ones((N,1))].T)
        return transformed_A.T

    def registration_errorM(self, parameters, array_A, array_B): # N x 2 arrays
        """
        Compute the registration error between two sets of 2D points after applying an affine transformation.

        Parameters:
        ----------

        - parameters (array-like): Affine transformation parameters.
        - array_A (array-like): Source set of 2D points (N x 2 array).
        - array_B (array-like): Target set of 2D points (N x 2 array).

        Returns:
        ----------

        - float: Registration error, calculated as the Frobenius norm of the difference
                between the transformed source points and the target points.
        """
        # Compute the registration error (sum of squared differences)
        transformed_A = self.affine_transformM(parameters, array_A)
        N = array_A.shape[0]
        error = (1/N)*np.linalg.norm(transformed_A - array_B,'fro')
        return error



    def optimize_affine_transform(self, Zc, B, Sc, ind_):
        """
        Optimize the parameters for the affine transformation.

        Parameters:
        ----------
        - Zc (array-like): Cluster data points (N x 2 array).
        - B (array-like): Matrix of vectors (2 x P) representing the original basis.
        - Sc (array-like): Singular values of the original basis.
        - ind_ (array-like): Boolean array indicating the indices of the cluster.

        Returns:
        --------
        - Tuple: A tuple containing the optimized parameters and the transformed cluster points.

        Notes:
        ------
        This function performs optimization to find the best affine transformation parameters
        using the Nelder-Mead method. It then applies the optimized transformation to the cluster points.
        """
        # Initial guess for optimization parameters (scale, rotation, translation)
        initial_parameters = np.array([1.0, 0.0, 0.0, 0.0])

        # Perform the optimization to find the best affine transformation
        result = minimize(self.registration_errorM, initial_parameters,
                          args=(Zc, self.Z[ind_]), method='Nelder-Mead')

        # Get the optimized parameters
        optimized_parameters = result.x

        # Apply the optimized transformation to the cluster points and vectors
        Zc_transformed = self.affine_transformM(optimized_parameters, Zc)
        B_transformed = self.affine_transformM(optimized_parameters, B)



        return optimized_parameters, Zc_transformed, B_transformed


    def clustering(self, Z, eps_=None, per_=5): #N x 2 array
        """
        Perform clustering on the given 2D data using DBSCAN algorithm.

        Parameters:
        ----------
        - Z (array-like): N x 2 list | np.ndarray  representing the data points.
        - eps_ (float, optional): The maximum distance between two samples for one to be considered
          as in the neighborhood of the other. Defaults to None.
        - per_ (float, optional): The percentile value used to set the `eps` parameter if it is not provided.
          Defaults to 5.

        Returns:
        --------
        - list | np.ndarray : An array of cluster labels assigned by the DBSCAN algorithm.

        Notes:
        ------
        If `eps_` is not provided, it is calculated as a percentile of the pairwise Euclidean distances
        between points in the input data `Z`.

        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm
        that groups together data points that are close to each other and marks outliers as noise.

        """
        if eps_ == None:
          eps_=np.percentile(squareform(cdist(Z, Z)),q=per_)
        np.random.seed(123)
        clus = DBSCAN(eps=eps_)
        clus.fit(Z)
        return clus.labels_

    def pca_by_SVD(self, X):
        """
        Perform SVD decomposition.

        Parameters:
        ----------
        - X: list | np.ndarray
          Input data N x P.

        Returns:
        --------

        -  U, S, VT, S_, A, B

        Details:
        --------

        Singular Value Decomposition

        (utilizar ..math:: en lugar de $$)
        $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^\top = \mathbf{U}\mathbf{S}^{0.5}\mathbf{S}^{0.5}\mathbf{V}^\top = \mathbf{A}\mathbf{B}^\top$

        $\mathbf{X}\in \mathbb{R}^{N \times P}$

        $\mathbf{U}\in \mathbb{R}^{N \times M}$

        $\mathbf{V}\in \mathbb{R}^{P \times M}$

        $\mathbf{S}\in \mathbb{R}^{M \times M}$

        $\mathbf{A} =  \mathbf{U}\mathbf{S}^{0.5} \in \mathbb{R}^{N \times M} $

        $\mathbf{B} = \mathbf{V}\mathbf{S}^{0.5} \in \mathbb{R}^{P \times M} $

        $M = min(N,P)$
        """
        #centering data
        X -= X.mean()



        # SVD decomposition

        U, S, VT = np.linalg.svd(X)

        # Use the full set of singular values
        S_ = S
        A = U[:, :S_.shape[0]].dot((np.diag(S_))**(0.5))  # samples-based basis
        B = VT.T[:, :S_.shape[0]].dot((np.diag(S_))**(0.5)) # features-based basis


        Zc = X.dot(VT.T[:,:2]) # x x V^T


        # PCA projection from SVD -> without standard scaler or min-max scaler


        return U, S, VT, S_, A, B, Zc
    
    def compute_variance_ratio(self,Sc):
      """
      Compute eigenvalues, total variance, and explained variance ratio by principal component.

      Parameters:
      - Sc: Array of singular values from SVD.

      Returns:
      - explained_variance_ratio: Array of explained variance ratios.
      """
      # Compute eigenvalues
      eigenvalues = Sc ** 2
      total_variance = np.sum(eigenvalues)
      explained_variance_ratio = np.round((eigenvalues / total_variance) * 100, 2)

      return explained_variance_ratio
        


    def get_localbp_(self, tar_, Ck, databp):

        # Set colormap
        cmap = plt.cm.get_cmap('Paired')
        # Create colormap for clusters
        cmap_ = cmap(np.linspace(0, 1, Ck))
        fontsize = 20

        U = []
        Vh = []
        S = []

        # Create subplots
        fig3, ax3 = plt.subplots(2, Ck + 1, figsize=(12, 7))
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

        # Iterate over clusters
        for i, v in enumerate(np.unique(tar_)):
            ind_ = tar_ == v

            # Perform PCA by SVD
            Uc, Sc, Vhc, S_, A, B, Zc = self.pca_by_SVD(self.X_[ind_])

            # Store results
            U.append(Uc)
            S.append(Sc)
            Vh.append(Vhc[:2, :].T)

            # Compute affine transform
            optimized_params, ZcA, VA = self.optimize_affine_transform(Zc, (Vhc[:2, :].T).dot(np.diag(Sc[:2])**(0.5)),
                                                                        Sc, ind_)

            # Compute eigenvalues and explained variance ratio
            explained_variance_ratio = self.compute_variance_ratio(Sc)
            print(f"Explained variance ratio by principal component: {explained_variance_ratio}")

            # Plot non-linear local-Biplot SVD
            self.plot_transformed_clusters(ax1, self.Z[ind_], VA, cmap_[i])

            # Plot correlation of input data
            sns.heatmap(np.abs(np.corrcoef(self.X_.T).round(3)),  ax=ax3[0, 0], vmin=0, vmax=1, cmap='Reds', cbar=False,
                        linecolor="w", linewidths=1, xticklabels=databp.columns, yticklabels=databp.columns)
            ax3[0, 0].set_ylabel('Input data', fontsize=fontsize)
            ax3[0, 0].tick_params(axis='y', labelsize=fontsize)
            ax3[0, 0].tick_params(axis='x', labelsize=fontsize)
            ax3[1, 0].axis("off")

            # Plot correlation biplot matrix B
            sns.heatmap(np.abs(np.corrcoef(Vhc.T[:, :Sc.shape[0]].dot((np.diag(Sc))**(0.5))).round(3)),  ax=ax3[1, i + 1],
                        vmin=0, vmax=1, robust=True, cmap='Reds', cbar=False, linecolor="w", linewidths=1,
                        yticklabels=databp.columns)

            # Plot correlation of input data matrix
            sns.heatmap(np.abs(np.corrcoef(self.X_[ind_].T).round(3)), vmin=0, vmax=1,  ax=ax3[0, i + 1], cmap='Reds',
                        cbar=False, linecolor="w", linewidths=1, xticklabels=False)
            ax3[0, i + 1].yaxis.set_ticklabels([])
            ax3[0, i + 1].set_title('Cluster ' + str(i + 1), fontsize=fontsize, color=cmap_[i])
            ax3[1, i + 1].set_xticks([])
            ax3[1, i + 1].yaxis.set_ticklabels([])

            if i == 1:
                ax3[1, i].set_ylabel('NLLBiplot', fontsize=fontsize)
                ax3[1, 1].tick_params(axis='x', labelsize=fontsize)
                ax3[1, 1].tick_params(axis='y', labelsize=fontsize)
                ax3[1, 1].xaxis.set_ticks(np.arange(len(databp.columns))+0.5)
                ax3[1, 1].xaxis.set_ticklabels(databp.columns)
                ax3[1, 1].yaxis.set_ticks(np.arange(len(databp.columns))+0.5)
                ax3[1, 1].yaxis.set_ticklabels(databp.columns)

        # Add colorbar for last heatmap
        cbar_ax = fig3.add_axes([0.93, 0.15, 0.015, 0.68])  
        cbar_ax.tick_params(labelsize=fontsize)
        norm = Normalize(vmin=0, vmax=1)  # Customize normalization if needed
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        fig3.colorbar(sm, cax=cbar_ax)
        plt.tight_layout()
        fig3.savefig('correlations_projections_and_clusters.pdf', bbox_inches='tight')
        fig1.savefig('local-biplot_SVD.pdf', bbox_inches='tight')
        plt.show()

    

        return

if __name__ == "__main__":
    rice_ = LocalBiplot(databp,labels= None,perplexity=None,red = 'tsne') #class instance
    rice_.LocalBiplot_()
