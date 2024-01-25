#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import f
from sklearn.decomposition import PCA
import scipy
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

class OneClassSIMCA:
    """
    Implements soft independed modelling of class analogies (SIMCA) for one target class.

    Parameters
    ----------
    n_components : int, optional
        Number of PCA components, by default 2.

    q : float, optional
        q-value: false discovery rate, by default 0.95.

    Attributes
    ---------
    n_components : int
        Parameter set during initialization.

    q : float
        Parameter set during initialization.

    pca : sklearn.decomposition.PCA
        The underlying PCA model.

    target : str
        Name of the target class. Parameter set with 'fit' method.

    Q_target : numpy.ndarray of shape (n_samples,)
        Q residuals of target class. Calculated in 'fit' method.

    Q_conf : float
        Q confidence limit of target class. Calculated in 'fit' method.

    Q_test : numpy.ndarray of shape (n_samples,)
        Q residuals of test samples. Calculated in 'predict' method.

    Tsq_target : numpy.ndarray of shape (n_samples,)
        T square values of target class. Calculated in 'fit' method.

    Tsq_conf : float
        T square confidence limit of target class. Calculated in 'fit' method.

    Tsq_test : numpy.ndarray of shape (n_samples,)
        T square of test samples. Calculated in 'predict' method.

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.utils import resample
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> # load example data and set target (class label)
    >>> X, y = load_iris(return_X_y=True)
    >>> target = 0
    >>>
    >>> # Find target data to train model and draw random test data
    >>> X_target = X[np.where(y == target)]
    >>> X_test, y_test = resample(X, y n_samples=50)
    >>>
    >>> # instantiate a model and fit target data
    >>> model = OneClassSIMCA(n_components=2, q=0.95)
    >>> model.fit(X_target, target)
    >>>
    >>> # make prediction on test data and calculate accuracy
    >>> y_pred = model.predict(X_test)
    >>> y_true = y_test == target
    >>> print(accuracy_score(y_true, y_pred))
    >>>
    >>> # visualize results
    >>> model.plot(hue=y_test)
    """

    def __init__(self, n_components=2, q=0.95):
        self.n_components = n_components
        self.q = q
        self.pca = PCA(n_components=n_components)
        self._fitted = False
        self._validated = False

    def fit(self, X_target, target):
        """
        Fit SIMCA model with data from target class.

        Parameters
        ----------
        X_target : numpy.ndarray of shape (n_samples, n_features)
            Training data for target class.

        target : str
            Name of the target class as identifier of the model and for plotting.
        """
        # set target as identifier and title in plot method
        self.target = target
        self.pca.fit(X_target)

        self.scores_target = self.pca.transform(X_target)
        self.loadings = self.pca.components_
        self.residuals_target = X_target - self.pca.inverse_transform(self.scores_target)

        self.Q_target = np.sum(self.residuals_target**2, axis=1)
        
        lambda_values = PCA().fit(X_target).explained_variance_[self.n_components:]
        theta1 = np.sum(lambda_values)
        theta2 = np.sum(lambda_values**2)
        theta3 = np.sum(lambda_values**3)
        h0 = 1 - 2 * theta1 * theta3 / (3 * theta2**2)
        calpha = scipy.stats.norm.ppf(self.q)
        fraction1 = calpha * np.sqrt(2 * theta2 * h0**2) / theta1
        fraction2 = theta2*h0*(h0-1) / theta1**2
        self.Q_conf = theta1 * (1 + fraction1 + fraction2)**(1/h0)

        self.Tsq_target = np.sum(
            (self.scores_target / np.std(self.scores_target, axis=0)) ** 2, axis=1
        )
        self.Tsq_conf = (
            f.ppf(q=0.95, dfn=self.n_components, dfd=self.scores_target.shape[0])
            * self.n_components
            * (self.scores_target.shape[0] - 1)
            / (self.scores_target.shape[0] - self.n_components)
        )

        self._fitted = True
        
        return self

    def predict(self, X_test, decision_rule="both"):
        """
        Applies fitted SIMCA model to test data and makes a prediction
        about the target class membership.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_samples, n_features)
            Test data as feature matrix.

        decision_rule : str, optional
            Prediction based on either 'Q', 'Tsq' or 'both',
            by default "both".

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Boolean result array.

        Raises
        ------
        ValueError
            If invalid value is given for decision_rule argument.
        """
        # set as attributes for plot method
        
        self.scores_test = self.pca.transform(X_test)
        self.residuals_test = X_test - self.pca.inverse_transform(self.scores_test)
        self.Q_test = np.sum(self.residuals_test**2, axis=1)
        self.Tsq_test = np.sum(
            (self.scores_test / np.std(self.scores_test, axis=0)) ** 2, axis=1
        )

        pred_Q = self.Q_test < self.Q_conf
        pred_Tsq = self.Tsq_test < self.Tsq_conf
        self._validated = True

        if decision_rule == "both":
            return np.logical_and(pred_Q, pred_Tsq)
        elif decision_rule == "Q":
            return pred_Q
        elif decision_rule == "Tsq":
            return pred_Tsq
        else:
            raise ValueError("Invalid decision method, use 'Q', 'Tsq' or 'both'.")

    def Tsq_Q_plot(self, hue=None, annotate=None):
        """
        Visualizes fitted SIMCA model as T square Q plot.
        Test data is included when the 'predict' method was used prior.

        Parameters
        ----------
        hue : iterable, optional
            Iterable with test data class labels to color markers by class.
            Only labels for test data is needed, labels for target class are already known.
            Ignores if 'predict' method was not used prior,
            by default None.

        annotate : iterable, optional
            Iterable with sample names to annotate markers with,
            by default None.

        Returns
        -------
        matplotlib.pyplot.axes

        Raises
        ------
        ValueError
            If instance was not fitted before calling 'plot'.
        """
        if not self._fitted:
            raise ValueError(
                "This model is not fitted yet! Call 'fit' with appropriate arguments before plotting."
            )

        if self._validated:
            X = np.concatenate((self.Q_target, self.Q_test))
            Y = np.concatenate((self.Tsq_target, self.Tsq_test))
            style = ["Training"] * len(self.Q_target) + ["Validation"] * len(
                self.Q_test
            )
            if hue is not None:
                hue = [self.target] * len(self.Q_target) + list(hue)
        else:
            X = self.Q_target
            Y = self.Tsq_target
            style = ["Training"] * len(self.Q_target)
            hue = None

        ax = sns.scatterplot(x=X, y=Y, style=style, hue=hue)

        if annotate is not None:
            for x, y, name in zip(X, Y, annotate):
                ax.text(x, y, name)

        if self._validated:
            ax.legend()

        ax.axhline(self.Tsq_conf, c="tab:red", linestyle=":")
        ax.axvline(self.Q_conf, c="tab:red", linestyle=":")
        ax.set_xlabel("Q")
        ax.set_ylabel("$T^2$", rotation=0, labelpad=15)
        ax.set_title(f"SIMCA model for target: {self.target}")
        return ax
    
    def scores_plot(self, y_train, y_test, x_comp=1, y_comp=2):
        """
        Visualizes scores of fitted SIMCA model as Scores Plot.

        Parameters:
        ---------- 
        y_train : numpy.array of shape(n_samples,)
            True class labels for training data.    
        
        y_test : numpy.array of shape(n_samples,)
            True class labels for test data.
            
        x_comp : int, optional
            Component x axis, by default 1.

        y_comp : int, optional
            Component y axis, by default 2.
            
        Returns
        -------
        matplotlib.pyplot.axes         
        """        
        self.y_train = y_train
        self.y_test = y_test
        
        if not self._fitted:
            raise ValueError(
                "This model is not fitted yet! Call 'fit' with appropriate arguments before plotting."
            )

        if self._validated:
            X = np.concatenate(
                (self.scores_target[:, x_comp - 1], self.scores_test[:, x_comp - 1])
            )
            Y = np.concatenate(
                (self.scores_target[:, y_comp - 1], self.scores_test[:, y_comp - 1])
            )
            hue = list(self.y_train) + list(self.y_test)
            style = ["Training"] * len(self.y_train) + ["Validation"] * len(
                self.y_test
            )

        ax = sns.scatterplot(x=X, y=Y, hue=hue, style=style)

        ax.legend()

        plt.xlabel(f"PC {x_comp}")
        plt.ylabel(f"PC {y_comp}")

        return ax
    
    def plot_loadings(self, dataset, PC=1, color_range=0.1, width=6, height=6):
        """
        Plots loadings of a principle component with the original retention
        and drift time coordinates.

        Parameters
        ----------
        dataset : ims.Dataset
            The dataset is needed for the retention and drift time
            coordinates. 
        
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
        ret_time = dataset[0].ret_time
        drift_time = dataset[0].drift_time

        loading_pc = self.loadings[PC - 1, :].reshape(len(ret_time), len(drift_time))

        expl_var = []
        for i in range(1, self.n_components + 1):
            expl_var.append(round(self.pca.explained_variance_ratio_[i - 1] * 100, 1))

        _, ax = plt.subplots(figsize=(width, height))

        plt.imshow(
            loading_pc,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=(-color_range),
            vmax=color_range,
            extent=(min(drift_time), max(drift_time), min(ret_time), max(ret_time)),
        )

        plt.colorbar().set_label("PCA loadings")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        plt.xlabel(dataset[0]._drift_time_label)
        plt.ylabel("Retention time [s]")
        plt.title(f"PCA loadings of PC {PC} ({expl_var[PC-1]} % of variance)")
        return ax




class MultiClassSIMCA:
    """
    Builds soft independent modelling of class analogies model out off OneClassSIMCA models for each class.

    Parameters
    ----------
    models : list
        OneClassSIMCA model per class.

    Attributes
    ---------
    nclasses : int
        Number of OneClasSIMCA models in models.
        
    classnames : list
        Names of targets in OneClasSIMCA models.
        
    res_df : pd.DataFrame of shape (n_targets, n_samples)
        Binary representation of classification.
        Calculated in the predict method.

    Raises
    ------
    ValueError
        If instance was not fitted in 'OneClassSimca' class.

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.utils import resample
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> # load example data
    >>> X, y = load_iris(return_X_y=True)
    >>>
    >>> # instantiate a model and fit target data
    >>> model = MultiClassSIMCA()
    >>> model.fit(X, y, n_components=2, q=0.95)
    >>>
    >>> # make prediction on test data and calculate accuracy
    >>> y_pred = model.predict(X_test)
    >>> y_true = y_test == target
    >>> print(accuracy_score(y_true, y_pred))
    >>>
    >>> # visualize results
    >>> model.plot()
    """
    
    def __init__(self, models=[]):
        self.models = models
        self.nclasses = len(models)
        self.classnames = [list(model.target) for model in models]
        for model in models:
            if not model._fitted:
                raise ValueError(
                f"OneClassSIMCA {model.target} is not fitted yet! Call 'fit' in OneClassSIMCA with appropriate arguments."
                ) 
            
    def fit(self, X_train, y_train, n_components=2, p=0.95):
        """
        Fit OneClasSIMCA models with training data.
        
        Parameters
        ----------
        X_train : numpy.ndarray of shape (n_samples, n_features)
            Training data as feature matrix.
        
        y_train : numpy.array of shape(n_samples,)
            True class labels for training data.
            
        n_components : int, optional
            Number of PCA components, by default 2.

        q : float, optional
            False discovery rate, by default 0.95.
        """
        
        self.models = []
        target_list = np.unique(y_train)
        for target in target_list:
            X_target = X_train[np.where(y_train == target)]
            model = OneClassSIMCA(n_components, p)
            model.fit(X_target, target)
            self.models.append(model)
            self.classnames.append(target)
            
        return self

    def predict(self, X_test, y_test=None):
        """
        Applies fitted SIMCA models to test data and makes a prediction
        about the class memberships.

        Parameters
        ----------
        X_test : numpy.ndarray of shape (n_samples, n_features)
            Test data as feature matrix.
            
        y_test : numpy.array of shape(n_samples,), optional
            True class labels for test data, by default None

        Returns
        -------
        pd.DataFrame of shape (n_targets, n_samples)
        """
        
        y_pred_class = []
        for model in self.models:
            y_pred = model.predict(X_test)
            y_pred_class.append(y_pred.astype('int'))
            
        self.res_df = pd.DataFrame(
            data=np.array(y_pred_class).T,
            columns=[model.target for model in self.models]
        )
        
        if y_test is not None:
            self.res_df.insert(loc=0, column="Sample name", value=list(y_test))
            
        return self.res_df
    
    def plot(self):
        
        """Visualizes the classification of test data.

        Returns
        -------
        matplotlib.pyplot.axes
        """
        
        self.res_df = self.res_df.set_index("Sample name").sort_index()
        
        # Define colors
        colors = ["tab:grey", "tab:cyan"]
        cmap = mcolors.LinearSegmentedColormap.from_list('Custom', colors, len(colors))

        ax = sns.heatmap(self.res_df, cmap=cmap)

        # Set the colorbar labels
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25,0.75])
        colorbar.set_ticklabels(['False', 'True'])
        
        ax.set(xlabel="Predicted label", ylabel="Sample name")
        
        return ax
    