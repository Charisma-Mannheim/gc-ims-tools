import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import CenteredNorm
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from ims.plsda import _vip_scores


class PLSR:

    def __init__(self, dataset, n_components=10, scale=True, **kwargs):
        self.dataset = dataset
        self.n_components = n_components
        self.scale = scale
        self._sk_pls = PLSRegression(n_components=n_components, scale=scale, **kwargs)

    def fit(self, X_train, y_train):
        self._sk_pls.fit(X_train, y_train)
        
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

        self.y_pred_train = self._sk_pls.predict(X_train)
        self.y_train = y_train

    def predict(self, X_test, y_test=None):
        y_pred = self._sk_pls.predict(X_test)
        if y_test is not None:
            self.rmse = round(mean_squared_error(y_test, y_pred, squared=False), 2)
            self.r2_score = round(r2_score(y_pred, y_test), 2)
            self.y_pred_test = y_pred
            self.y_test = y_test
        return y_pred

    def calc_vip_scores(self, threshold=None):
        """
        Calculates variable importance in projection (VIP) scores.
        Optionally only of features with high coefficients.

        Parameters
        ----------
        top_n_coeff : int, optional
            Number highest coefficients per group
            if None calculates all, by default None
            
        threshold : int, optional
            if given keeps only VIP scores greater than threshold,
            by default None

        Returns
        -------
        numpy.ndarray
        """
        vips = _vip_scores(self.x_weights, self.x_scores, self.y_loadings)

        if threshold is not None:
            vip_array = np.zeros_like(vips)
            i = np.where(vips > threshold)
            vip_array[i] = vips[i]
            vips = vip_array

        self.vip_scores = vips
        return vips
    
    def plot(self):
        """
        Plots prediction vs actual values and shows regression line.

        Returns
        -------
        matplotlib.pyplot.axes
        """        
        z = np.polyfit(self.y_train, self.y_pred_train, 1)

        _, ax = plt.subplots(figsize=(9, 8))
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

    def plot_loadings(self, component=1, color_range=0.01):
        """
        Plots PLS x loadings as image with retention and drift
        time coordinates.

        Parameters
        ----------
        component : int, optional
            Component to plot, by default 1

        color_range : float, optional
            Minimum and Maximum to adjust to different scaling methods,
            by default 0.02

        Returns
        -------
        matplotlib.pyplot.axes
        """        
        loadings = self.x_loadings[:, component-1].\
            reshape(self.dataset[0].shape)
            
        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        _, ax = plt.subplots(figsize=(9, 10))

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
    
    def plot_coefficients(self):
        """
        Plots PLS coefficients as image with retention and drift time axis.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        coef = self.coefficients.reshape(self.dataset[0].values.shape)

        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time
        
        _, ax = plt.subplots(figsize=(9, 10))
        
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
    
    def plot_vip_scores(self):
        """
        Plots VIP scores as image with retention and drift time axis.

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

        _, ax = plt.subplots(figsize=(9, 10))

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
