import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import CenteredNorm
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score)


class PLS_DA:
    """
    PLS-DA classifier built using the scikit-learn PLSRegression implementation.
    Provides prebuilt plots and feature selection via variable importance in projection (VIP)
    scores.

    See the scikit-learn documentation for more details:
    https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

    Parameters
    ----------
    dataset : ims.Dataset
        Needed for the retention and drift time coordinates in the plots.

    n_components : int, optional
        Number of components to keep, by default 2.

    scale : bool, optional
        Wheather to scale X and y, by default True.

    kwargs : optional
        Additional key word arguments are passed on to the scikit-learn PLSRegression.

    Attributes
    ----------
    x_scores : numpy.ndarray of shape (n_samples, n_components)
        X scores.

    y_scores : numpy.ndarray of shape (n_samples, n_components)
        y scores.

    x_weights : numpy.ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each iteration.

    y_weights : numpy.ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each iteration.

    x_loadings : numpy.ndarray of shape (n_features, n_components)
        The loadings of X. When scaling was applied on the dataset,
        corrects the loadings using the weights.

    y_loadings : numpy.ndarray of shape (n_targes, n_components)
        The loadings of y.

    coefficients : numpy.ndarray of shape (n_features, n_targets)
        The coefficients of the linear model.
 
    y_pred_train : numpy.ndarray
        Stores the predicted values from the training data for the plot method.
        
    accuracy : float
        If y_test is set in predict method, calculates accuracy internally.
        
    precision : float
        If y_test is set in predict method, calculates precision internally.

    recall : float
        If y_test is set in predict method, calculates recall internally.

    Example
    -------
    >>> import ims
    >>> ds = ims.Dataset.read_mea("IMS_data")
    >>> X_train, X_test, y_train, y_test = ds.train_test_split()
    >>> model = ims.PLS_DA(ds, n_components=5)
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test, y_test)
    >>> model.plot()
    """
    def __init__(self, dataset, n_components=10, scale=True, **kwargs):
        self.dataset = dataset
        self.n_components = n_components
        self.scale = scale
        self._sk_pls = PLSRegression(n_components=n_components,
                                     scale=scale, **kwargs)

    @staticmethod
    def _create_binary_labels(y):
        """
        Creates a binary label matrix with one column per group.
        """
        groups = np.unique(y)
        y_binary = np.zeros((len(y), len(groups)))
        for i, j in enumerate(groups):
            col = [j in label for label in y]
            y_binary[:, i] = col
        return y_binary

    @staticmethod
    def _reverse_binary_labels(y, groups):
        """
        Returns a list of class labels from a binary label matrix and a tuple
        of unique groups.
        """
        y_reversed = []
        for label in y:
            i = np.argmax(label)
            y_reversed.append(groups[i])
        return y_reversed

    def fit(self, X_train, y_train):
        """
        Fits the model with training data.
        Converts the labels into a binary matrix.

        Parameters
        ----------
        X_train : numpy.ndarray of targets (n_samples, n_features)
            Training vectors with features.

        y_train : numpy.ndarray of shape (n_samples,)
            True class labels for training data.

        Returns
        -------
        self
            Fitted model.
        """
        self.groups = np.unique(y_train)
        self.y_train = y_train
        y_binary = self._create_binary_labels(y_train)
        self._sk_pls.fit(X_train, y_binary)
        # self.x_scores, self.y_scores = self._sk_pls.transform(X_train, y_train)
        self.x_scores = self._sk_pls.x_scores_
        self.y_scores = self._sk_pls.y_scores_
        self.x_weights = self._sk_pls.x_weights_
        self.y_weights = self._sk_pls.y_weights_
        self.y_loadings = self._sk_pls.y_loadings_
        self.coefficients = self._sk_pls.coef_

        if hasattr(self.dataset, "weights"):
            self.x_loadings = self._sk_pls.x_loadings_
        else:
            self.x_loadings = self._sk_pls.x_loadings_ / self.weights[:, None]

        return self

    def predict(self, X_test, y_test=None):
        """
        Predicts class labels for test data. Converts back from binary
        labels matrix to a list of class names. If y_test is set also calculates
        accuracy, precision and recall.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_samples, n_features)
            Feature vectors of test dataset.

        y_test : numpy.ndarray of shape (n_samples,), optional
            True labels for test dataset. If set calculates error metrics,
            by default None

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted class labels.
        """        
        y_pred_binary = self._sk_pls.predict(X_test)
        y_pred = self._reverse_binary_labels(y_pred_binary, self.groups)

        if y_test is not None:
            self.y_test = y_test
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred,
                                             average="weighted")
            self.recall = recall_score(y_test, y_pred,
                                       average="weighted")

        return np.array(y_pred)

    def calc_vip_scores(self, threshold=None):
        """
        Calculates variable importance in projection (VIP) scores.
        Setting a threshold results in less cluttered plots.

        Parameters
        ----------
        threshold : int or float, optional
            If set keeps only VIP scores greater than threshold
            and sets all other to zero, by default None.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
        """
        vips = _vip_scores(self.x_weights, self.x_scores, self.y_loadings)
        if threshold is not None:
            vip_array = np.zeros_like(vips)
            i = np.where(vips > threshold)
            vip_array[i] = vips[i]
            vips = vip_array

        self.vip_scores = vips
        return vips
            
    def plot(self, x_comp=1, y_comp=2, width=9, height=8,
             annotate=False):
        """
        Plots PLS components as scatter plot.

        Parameters
        ----------
        x_comp : int, optional
            Component x axis, by default 1.

        y_comp : int, optional
            Component y axis, by default 2.
            
        width : int or float, optional
            Width of the plot in inches,
            by default 9.

        height : int or float, optional
            Height of the plot in inches,
            by default 8.

        annotate : bool, optional
            If True adds sample names to markers,
            by default False.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        cols = [f"PLS Component {i}" for i in range(1, self.n_components + 1)]

        df = pd.DataFrame(self.x_scores, columns=cols)
        df["Group"] = self.y_train
        # df["Sample"] = self.dataset.samples
        
        plt.figure(figsize=(width, height))
        ax = sns.scatterplot(
            x=f"PLS Component {x_comp}",
            y=f"PLS Component {y_comp}",
            data=df,
            hue="Group",
            style="Group",
            s=50
            )
        plt.legend(frameon=True, fancybox=True, facecolor="white")
        
        # if annotate:
        #     for _, row in df.iterrows():
        #         plt.annotate(
        #             row["Sample"],
        #             xy=(row[f"PLS Component {x_comp}"], row[f"PLS Component {y_comp}"]),
        #             xycoords="data"
        #         )

        return ax
    
    def plot_loadings(self, component=1, color_range=0.02,
                      width=9, height=10):
        """
        Plots PLS x loadings as image with retention and drift
        time coordinates.

        Parameters
        ----------
        component : int, optional
            Component to plot, by default 1.

        color_range : float, optional
            Minimum and Maximum to adjust to different scaling methods,
            by default 0.02.
            
        width : int or float, optional
            Width of the plot in inches,
            by default 9.

        height : int or float, optional
            Height of the plot in inches,
            by default 10.

        Returns
        -------
        matplotlib.pyplot.axes
        """        
        loadings = self.x_loadings[:, component-1].\
            reshape(self.dataset[0].shape)
            
        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        _, ax = plt.subplots(figsize=(width, height))

        plt.imshow(
            loadings,
            cmap="RdBu_r",
            vmin=(-color_range),
            vmax=color_range,
            origin="lower",
            aspect="auto",
            extent=(min(drift_time), max(drift_time),
                    min(ret_time), max(ret_time))
            )

        plt.colorbar(label="Loadings")
        plt.title(f"PLS-DA loadings of component {component}")
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention Time [s]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax

    def plot_coefficients(self, group=0, width=9, height=10):
        """
        Plots PLS coefficients of selected group as image
        with retention and drift time axis.

        Parameters
        ----------
        group : int or str, optional
            Index or name of group, by default 0.

        width : int or float, optional
            Width of the plot in inches,
            by default 9.

        height : int or float, optional
            Height of the plot in inches,
            by default 10.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        if isinstance(group, str):
            group_index = self.groups.index(group)
            group_name = group

        if isinstance(group, int):
            group_index = group
            group_name = self.groups[group]

        coef = self._sk_pls.coef_[:, group_index].\
            reshape(self.dataset[0].values.shape)

        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time
        
        _, ax = plt.subplots(figsize=(width, height))
        
        plt.imshow(
            coef,
            cmap="RdBu_r",
            norm=CenteredNorm(0),
            origin="lower",
            aspect="auto",
            extent=(min(drift_time), max(drift_time),
                    min(ret_time), max(ret_time))
            )

        plt.colorbar(label="Coefficients")
        plt.title(f"PLS-DA coefficients of {group_name}")
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention Time [s]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax
    
    def plot_vip_scores(self, width=9, height=10):
        """
        Plots VIP scores as image with retention and drift time axis.
        
        Parameters
        ----------
        width : int or float, optional
            Width of the plot in inches,
            by default 9.

        height : int or float, optional
            Height of the plot in inches,
            by default 10.

        Returns
        -------
        matplotlib.pyplot.axes

        Raises
        ------
        ValueError
            If VIP scores have not been calculated prior.
        """        
        if not hasattr(self, "vip_scores"):
            raise ValueError("Must calculate VIP scores first.")
        
        vip_matrix = self.vip_scores.reshape(self.dataset[0].values.shape)

        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        _, ax = plt.subplots(figsize=(width, height))

        plt.imshow(
            vip_matrix,
            cmap="RdBu_r",
            origin="lower",
            aspect="auto",
            extent=(min(drift_time), max(drift_time),
                    min(ret_time), max(ret_time))
            )

        plt.colorbar(label="VIP scores")
        plt.title(f"PLS VIP scores")
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention Time [s]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax


def _vip_scores(xw, xs, yl):
    """
    Calculates variable importance in projection (VIP) scores
    from scikit-learn PLSRegression x_weights_, x_scores_, and y_loadings_.

    Parameters
    ----------
    xw : numpy.ndarray of shape (n_features, n_components)
        x_weights_

    xs : numpy.ndarray of shape (n_samples, n_components)
        x_scores_

    yl : numpy.ndarray of shape (n_targes, n_components)
        y_loadings_

    Returns
    -------
    numpy.ndarray of shape (n_features,)
        Vector of VIP scores.
    """
    W0 = xw / np.sqrt(np.sum(xw**2, 0))
    p, _ = xw.shape
    sum_sq = np.sum(xs**2, 0) * np.sum(yl**2, 0)
    vips = np.sqrt(p * np.sum(sum_sq * (W0**2), 1) / np.sum(sum_sq))
    return vips
