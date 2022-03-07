import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from sklearn.decomposition import PCA


class PCA_Model:
    """
    PCA_Model is a wrapper class around the scikit-learn PCA implementation
    and provides prebuilt plots for GC-IMS datasets.

    See the original scikit-learn documentation for a detailed description:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Parameters
    ----------
    dataset : ims.Dataset
        The dataset is needed for the retention and drift time
        coordinates.

    n_components : int or float, optional
        Number of components to keep. If None all components are kept,
        by default None.

    svd_solver : str, optional
        "auto", "full", "arpack" or "randomised" are valid,
        by default "auto".

    **kwargs: optional
        Additional key word arguments are passed to the scikit-learn PCA.
        See the original documentation for valid parameters.

    Attributes
    ----------
    scores : numpy.ndarray of shape (n_samples, n_features)
        X with dimension reduction applied.

    loadings : numpy.ndarray of shape (n_components, n_features)
        PCA loadings already corrected when a scaling method was
        applied on the dataset.

    explainded_variance : numpy.ndarray of shape (n_components,)
        The amount of variance explained by each component.

    explained_variance_ratio : numpy.nd_array of shape (n_components,)
        Percentage of variance explained by each component.

    singular_values : numpy.ndarray of shape (n_components,)
        The singular values corresponding to each component.

    mean : numpy.ndarray of shape (n_features,)
        Per feature mean estimated from training data.

    Example
    -------
    >>> import ims
    >>> ds = ims.Dataset.read_mea("IMS_data")
    >>> X, _ = ds.get_xy()
    >>> pca = ims.PCA_Model(ds, n_components=20)
    >>> pca.fit(X)
    >>> pca.plot()
    """
    def __init__(self, dataset, n_components=None, svd_solver="auto", **kwargs):
        self.dataset = dataset
        if n_components is None:
            self.n_components = len(self.dataset)
        else:
            self.n_components = n_components
        self.svd_solver = svd_solver
        self._sk_pca = PCA(n_components, svd_solver=svd_solver, **kwargs)

    def fit(self, X_train):
        """
        Fit the PCA model with training data.

        Parameters
        ----------
        X_train : numpy.ndarray of shape (n_samples, n_features)
            The training data.
            
        Returns
        -------
        self
            The fitted model.
        """
        self._sk_pca.fit(X_train)
        self.scores = self._sk_pca.transform(X_train)
        self.explained_variance = self._sk_pca.explained_variance_
        self.explained_variance_ratio = self._sk_pca.explained_variance_ratio_
        self.singular_values = self._sk_pca.singular_values_
        self.mean = self._sk_pca.mean_
        self.loadings = self._sk_pca.components_
        return self

    def plot(self, PC_x=1, PC_y=2, width=9, height=8, annotate=False):
        """
        Scatter plot of selected principal components.

        Parameters
        ----------
        PC_x : int, optional
            PC x axis, by default 1.

        PC_y : int, optional
            PC y axis, by default 2.

        width : int or float, optional
            plot width in inches, by default 8.

        height : int or float, optional
            plot height in inches, by default 7.
            
        annotate : bool, optional
            label data points with sample name,
            by default False.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        expl_var = []
        for i in range(1, self.n_components + 1):
            expl_var.append(round(self.explained_variance_ratio[i-1] * 100, 1))
        
        pc_df = pd.DataFrame(
            data=self.scores,
            columns=[f"PC {x}" for x in range(1, self.n_components + 1)]
        )
        pc_df['Sample'] = self.dataset.samples
        pc_df['Label'] = self.dataset.labels

        _, ax = plt.subplots(figsize=(width, height))
        sns.scatterplot(
            ax=ax,
            x=f"PC {PC_x}",
            y=f"PC {PC_y}",
            data=pc_df,
            hue="Label",
            style="Label",
        )

        plt.legend(frameon=True, fancybox=True, facecolor="white")
        plt.xlabel(f"PC {PC_x} ({expl_var[PC_x-1]} % of variance)")
        plt.ylabel(f"PC {PC_y} ({expl_var[PC_y-1]} % of variance)")

        if annotate:
            for i, point in pc_df.iterrows():
                ax.text(point[f"PC {PC_x}"], point[f"PC {PC_y}"],
                        point["Sample"])

        return ax

    def plot_loadings(self, PC=1, color_range=0.1, width=9, height=10):
        """
        Plots loadings of a principle component with the original retention
        and drift time coordinates.

        Parameters
        ----------
        PC : int, optional
            principal component, by default 1.

        color_range : int, optional
            color_scale ranges from - color_range to + color_range
            centered at 0.

        width : int or float, optional
            plot width in inches, by default 9.

        height : int or float, optional
            plot height in inches, by default 10.

        Returns
        -------
        matplotlib.pyplot.axes
        """        
        # use retention and drift time axis from the first spectrum
        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        loading_pc = self.loadings[PC-1, :].reshape(len(ret_time),
                                                    len(drift_time))
        
        expl_var = []
        for i in range(1, self.n_components + 1):
            expl_var.append(round(self.explained_variance_ratio[i-1] * 100, 1))

        _, ax = plt.subplots(figsize=(width, height))

        plt.imshow(
            loading_pc,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=(-color_range),
            vmax=color_range,
            extent=(min(drift_time), max(drift_time),
                    min(ret_time), max(ret_time))
            )

        plt.colorbar().set_label("PCA Loadings")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention Time [s]")
        plt.title(f"PCA Loadings of PC {PC} ({expl_var[PC-1]} % of variance)")
        return ax

    def scree_plot(self, width=9, height=8):
        """
        Plots the explained variance ratio per principal component
        and cumulatively.

        Parameters
        ----------
        width : int or float, optional
            Width of the plot in inches,
            by default 9.

        height : int or float, optional
            Height of the plot in inches,
            by default 8.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        x = [*range(1, self.n_components + 1)]
        y = self.explained_variance_ratio

        _, ax = plt.subplots(figsize=(width, height))
        
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(MaxNLocator(integer=True))
        
        plt.xticks(x)
        plt.xlabel("Principal Component")
        plt.ylabel("Explainded variance ratio [%]")
        
        ax.plot(x, np.cumsum(y) * 100, label="cumulative")
        ax.plot(x, y * 100, label="per PC")
        
        plt.legend(frameon=True, fancybox=True, facecolor="white")
        return ax
