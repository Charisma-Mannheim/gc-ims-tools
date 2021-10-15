import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import CenteredNorm
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from ims.plsda import _vip_scores


class PLSR:
    """
    Applies a scikit-learn PLSRegression to GC-IMS data and provides prebuilt
    plots as well as a feature selection via variable importance in projection (VIP)
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
        
    rmse : float
        If y_test is set in predict method calculates root mean squared error.
        
    r2_score : float
        If y_test is set in predict method calculates R^2 score.

    Example
    -------
    >>> import ims
    >>> import pandas as pd
    >>> ds = ims.Dataset.read_mea("IMS_data")
    >>> responses = pd.read_csv("responses.csv")
    >>> ds.labels = responses
    >>> X_train, X_test, y_train, y_test = ds.train_test_split()
    >>> model = ims.PLSR(ds, n_components=5)
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test, y_test)
    >>> model.plot()
    """
    def __init__(self, dataset, n_components=2, scale=True, **kwargs):
        self.dataset = dataset
        self.n_components = n_components
        self.scale = scale
        self._sk_pls = PLSRegression(n_components=n_components, scale=scale, **kwargs)

    def fit(self, X_train, y_train):
        """
        Fits the model with training data.

        Parameters
        ----------
        X_train : numpy.ndarray of targets (n_samples, n_features)
            Training vectors with features.

        y_train : numpy.ndarray of shape (n_samples, n_targets)
            Target vectors with response variables.

        Returns
        -------
        self
            Fitted model.
        """
        self._sk_pls.fit(X_train, y_train)
        self.x_scores, self.y_scores = self._sk_pls.transform(X_train, y_train)
        self.x_weights = self._sk_pls.x_weights_
        self.y_weights = self._sk_pls.y_weights_
        self.y_loadings = self._sk_pls.y_loadings_
        self.coefficients = self._sk_pls.coef_

        if hasattr(self.dataset, "weights"):
            self.x_loadings = self._sk_pls.x_loadings_ / self.dataset.weights[:, None]
        else:
            self.x_loadings = self._sk_pls.x_loadings_

        self.y_pred_train = self._sk_pls.predict(X_train)
        self.y_train = y_train
        return self

    def predict(self, X_test, y_test=None):
        """
        Predicts responses for features of the test data.
        If y_test is set calculates root mean squared error and
        R^2 score.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_samples, n_features)
            Features of test data.

        y_test : numpy.ndarray of shape (n_samples, n_targets), optional
            Target vector of response variables. If set calculates error metrics,
            by default None.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_targets)
            Predicted responses for test data.
        """
        y_pred = self._sk_pls.predict(X_test)
        if y_test is not None:
            self.rmse = round(mean_squared_error(y_test, y_pred, squared=False), 2)
            self.r2_score = round(r2_score(y_pred, y_test), 2)
            self.y_pred_test = y_pred
            self.y_test = y_test
        return y_pred
    
    def score(self, X_test, y_test, sample_weight=None):
        """
        Calculates R^2 score score for predicted data.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_samples, n_features)
            Feature vectors of the test data.

        y_test : numpy.ndarray of shape (n_samples, n_targets)
            True regression responses.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights, by default None.

        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred, sample_weight=sample_weight)

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

    def plot(self, width=9, height=8):
        """
        Plots predicted vs actual values and shows regression line.
        Recommended to predict with test data first.

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
        z = np.polyfit(self.y_train, self.y_pred_train, 1)

        _, ax = plt.subplots(figsize=(width, height))
        plt.scatter(self.y_pred_train, self.y_train)

        if hasattr(self, "y_pred_test"):
            plt.scatter(self.y_pred_test, self.y_test, c="tab:orange")
            y = self.y_test
            label = f"RMSE: {self.rmse}"
        else:
            y = self.y_train
            label = None

        plt.plot(
            np.polyval(z, y),
            y,
            label=label,
            c="tab:orange",
            linewidth=1
            )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.legend(frameon=True, fancybox=True, facecolor="white")
        return ax

    def plot_loadings(self, component=1, color_range=0.01, width=9, height=10):
        """
        Plots PLS x loadings as image with retention and drift
        time coordinates.

        Parameters
        ----------
        component : int, optional
            Component to plot, by default 1.

        color_range : float, optional
            Minimum and maximum to adjust to different scaling methods,
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
        plt.title(f"PLS loadings of component {component}")
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention Time [s]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax
    
    def plot_coefficients(self, width=9, height=10):
        """
        Plots PLS coefficients as image with retention and drift time axis.

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
        """
        coef = self.coefficients.reshape(self.dataset[0].values.shape)

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
        plt.title("PLS Coefficients")
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
