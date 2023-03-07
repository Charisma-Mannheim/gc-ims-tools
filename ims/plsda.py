import ims
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import CenteredNorm
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score


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
        
    vip_scores : numpy.ndarray of shape (n_features,)
        Variable importance in projection (VIP) scores.
 
    y_pred_train : numpy.ndarray
        Stores the predicted values from the training data for the plot method.

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
    def __init__(self, dataset, n_components=2, **kwargs):
        self.dataset = dataset
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components,
                                 scale=False, **kwargs)
        self._binarizer = LabelBinarizer()
        self._fitted = False
        self._validated = False

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
        y_binary = self._binarizer.fit_transform(y_train)
        self.pls.fit(X_train, y_binary)
        self.x_scores, self.y_scores = self.pls.transform(X_train, y_binary)
        self.x_weights = self.pls.x_weights_
        self.x_loadings = self.pls.x_loadings_
        self.y_weights = self.pls.y_weights_
        self.y_loadings = self.pls.y_loadings_
        self.coefficients = self.pls.coef_
        self.vip_scores = ims.utils.vip_scores(self.x_weights, self.x_scores,
                                               self.y_loadings)
        self._fitted = True
        return self

    def predict(self, X_test):
        """
        Predicts class labels for test data. Converts back from binary
        labels matrix to a list of class names. If y_test is set also calculates
        accuracy, precision and recall and stores them as attributes.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_samples, n_features)
            Feature vectors of test dataset.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        y_pred = self.pls.predict(X_test)
        y_pred = self._binarizer.inverse_transform(y_pred)
        self.x_scores_pred = self.pls.transform(X_test)
        self._validated = True
        return y_pred

    def transform(self, X, y=None):
        """
        Apply the dimensionality reduction.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.

        y : numpy.ndarray of shape (n_samples, n_targtets), optional
            Dependend variables, by default None

        Returns
        -------
        tuple
            X_scores
        """
        return self.pls.transform(X, y)
    
    def score(self, X_test, y_test, sample_weight=None):
        """
        Calculates accuracy score for predicted data.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_samples, n_features)
            Feature vectors of the test data.

        y_test : numpy.ndarray of shape (n_samples,)
            True classification labels.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights, by default None.

        Returns
        -------
        score : float
            Mean accuracy score.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred, sample_weight=sample_weight)

    def plot(self, x_comp=1, y_comp=2, annotate=False):
        """
        Plots PLS components as scatter plot.

        Parameters
        ----------
        x_comp : int, optional
            Component x axis, by default 1.

        y_comp : int, optional
            Component y axis, by default 2.

        annotate : bool, optional
            If True adds sample names to markers,
            by default False.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        if not self._fitted:
            raise ValueError(
                "This model is not fitted yet! Call 'fit' with appropriate arguments before plotting."
            )

        if self._validated:
            X = np.concatenate((self.x_scores[:, x_comp-1],
                                self.x_scores_pred[:, x_comp-1]))
            Y = np.concatenate((self.x_scores[:, y_comp-1],
                                self.x_scores_pred[:, y_comp-1]))
            hue = list(self.y_train)\
                + self.dataset[self.dataset.test_index].labels
            style = ["Training"] * len(self.y_train)\
                + ["Validation"] * len(self.dataset.test_index)
            if hasattr(self.dataset, "train_index"):
                sample_names = self.dataset[self.dataset.train_index].samples\
                    + self.dataset[self.dataset.test_index].samples
            else:
                sample_names = self.dataset.samples

        else:
            X = self.x_scores[:, x_comp-1]
            Y = self.x_scores[:, y_comp-1]
            hue = self.y_train
            style = self.y_train
            sample_names = self.y_train

        ax = sns.scatterplot(x=X, y=Y, hue=hue, style=style)

        ax.legend()

        plt.xlabel(f"Latent variable {x_comp}")
        plt.ylabel(f"Latent variable {y_comp}")

        if annotate:
            for x, y, name in zip(X, Y, sample_names):
                ax.text(x, y, name)

        return ax

    def plot_loadings(self, component=1, color_range=0.02,
                      width=8, height=8):
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
            by default 8.

        height : int or float, optional
            Height of the plot in inches,
            by default 8.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        if not self._fitted:
            raise ValueError(
                "This model is not fitted yet! Call 'fit' with appropriate arguments before plotting."
            )

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

        plt.colorbar(label="PLS-DA loadings")
        plt.title(f"PLS-DA loadings of component {component}")
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention time [s]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax

    def plot_coefficients(self, group=0, width=8, height=8):
        """
        Plots PLS coefficients of selected group as image
        with retention and drift time axis.

        Parameters
        ----------
        group : int or str, optional
            Index or name of group, by default 0.

        width : int or float, optional
            Width of the plot in inches,
            by default 8.

        height : int or float, optional
            Height of the plot in inches,
            by default 8.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        if not self._fitted:
            raise ValueError(
                "This model is not fitted yet! Call 'fit' with appropriate arguments before plotting."
            )

        if isinstance(group, str):
            group_index = self.groups.index(group)
            group_name = group

        if isinstance(group, int):
            group_index = group
            group_name = self.groups[group]

        coef = self.pls.coef_[:, group_index].\
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

        plt.colorbar(label="PLS-DA coefficients")
        plt.title(f"PLS-DA coefficients of {group_name}")
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention time [s]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax
    
    def plot_vip_scores(self, threshold=None, width=8, height=8):
        """
        Plots VIP scores as image with retention and drift time axis.
        
        Parameters
        ----------
        threshold : int
            Only plots VIP scores above threshold if set.
            Values below are displayed as 0,
            by default None.

        width : int or float, optional
            Width of the plot in inches,
            by default 8.

        height : int or float, optional
            Height of the plot in inches,
            by default 8.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        if not self._fitted:
            raise ValueError(
                "This model is not fitted yet! Call 'fit' with appropriate arguments before plotting."
            )
        if threshold is None:
            vip_matrix = self.vip_scores.reshape(self.dataset[0].values.shape)
        else:
            vips = np.zeros_like(self.vip_scores)
            i = np.where(self.vip_scores > threshold)
            vips[i] = self.vip_scores[i]
            vip_matrix = vips.reshape(self.dataset[0].values.shape)

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
        plt.title(f"PLS-DA VIP scores")
        plt.xlabel(self.dataset[0]._drift_time_label)
        plt.ylabel("Retention time [s]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax
