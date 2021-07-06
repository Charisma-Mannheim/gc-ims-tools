import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score
)
from ims import BaseModel


class PLSR(BaseModel):
    
    def __init__(self, dataset, scaling_method=None, optimize=True,
                 n_components=20, kfold=5, **kwargs):
        
        super().__init__(dataset, scaling_method)
        self.kfold = kfold
        self.n_components = n_components
        self.optimize = optimize

        if self.optimize:
            # _accuracies is needed for the optimization plot
            self._best_comp, self._accuracies = self._optimize_pls()

        self._pls = PLSRegression(n_components=self._best_comp, **kwargs)
        self._pls.fit(self.X, self.y)
        
        self.prediction = self._pls.predict(self.X)
        self.r2_score = round(r2_score(self.y, self.prediction), 3)
        self.accuracy = round(mean_squared_error(self.y, self.prediction), 3)
        
        self.prediction_cv = cross_val_predict(self._pls, self.X, self.y, cv=kfold)
        self.r2_score_cv = round(r2_score(self.y, self.prediction_cv), 3)
        self.mse_cv = round(mean_squared_error(self.y, self.prediction_cv), 3)
        
        self.result = {
            "r2 score": self.r2_score,
            "accuracy": self.accuracy,
            "r2 score cv": self.r2_score_cv,
            "accuracy cv": self.mse_cv
        }

    def _optimize_pls(self):
        accuracy = []
        component = np.arange(1, self.n_components)
        
        for i in component:
            pls = PLSRegression(n_components=i)
            y_cv = cross_val_predict(pls, self.X, self.y, cv=self.kfold)
            accuracy.append(mean_squared_error(self.y, y_cv))
            
        best_ac = np.argmin(accuracy)
        return component[best_ac], accuracy
        
    def plot_optimization(self):
        component = np.arange(1, self.n_components)
        best_ac = np.argmin(self._accuracies)

        with plt.style.context("seaborn"):
            fig = plt.figure()
            plt.plot(component, self._accuracies)
            plt.scatter(component, self._accuracies)
            plt.plot(component[best_ac], self._accuracies[best_ac], color="tab:orange",
                        marker="*", markersize=20)
            plt.xlabel("Number of PLS Components")
            plt.ylabel("MSE")
            plt.title("PLS")
        return fig

    def plot(self):
        z = np.polyfit(self.y, self.prediction, 1)
        with plt.style.context("seaborn"):
            fig = plt.figure()
            plt.scatter(self.prediction, self.y)
            plt.plot(self.y, self.y, label="Ideal", c="tab:green", linewidth=1)
            plt.plot(np.polyval(z, self.y), self.y, label="Regression",
                     c="tab:orange", linewidth=1)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Crossvalidation")
            plt.legend(frameon=True, fancybox=True, facecolor="white")
        return fig


class PLS_DA(BaseModel):

    def __init__(self, dataset, scaling_method=None, optimize=False,
                 n_components=10, kfold=5):
        """
        Performs PLS_DA with IMS data. Automatically determines the optimal
        number of components.

        Parameters
        ----------
        dataset : ims.Dataset

        scaling_method : str, optional
            'standard', 'auto', 'pareto' and 'var' are valid arguments,
            by default None

        optimize : bool, optional
            If true finds the number of components with highest accuracy, by default False

        n_components : int, optional
            If optimize is true the maximum number of components
            otherwise the number of components for PLS, by default 10

        kfold : int, optional
            Number of splits for crossvalidation, by default 5
        """        
        super().__init__(dataset, scaling_method)
        self.optimize = optimize
        self.n_components = n_components
        self.kfold = kfold

        # generates the binary matrix used as labels for PLS regression
        # one column for each group and one row for each sample
        self.groups = list(np.unique(self.dataset.labels))
        self.y_binary = np.zeros((len(self.dataset), len(self.groups)))
        for i, j in enumerate(self.groups):
            col = [j in label for label in self.dataset.labels]
            self.y_binary[:, i] = col

        # either iterates from 1 to n_components and finds best accuracy
        # or uses n_components as the number of components directly 
        if self.optimize:
            self._best_comp, self._accuracies = self._optimize_plsda()
            self._fit(self._best_comp)
            self.accuracy, self.precision, self.recall = self._crossval(self._best_comp)
        else:    
            self._fit(self.n_components)
            self.accuracy, self.precision, self.recall = self._crossval(self.n_components)

    def _crossval(self, n):
        '''Crossvalidation zu optimize number of components'''
        kf = KFold(self.kfold, shuffle=True, random_state=1)

        accuracy = []
        precision = []
        recall = []
        for train_index, test_index in kf.split(self.X):
            X_train = self.X[train_index, :]
            y_train = self.y_binary[train_index, :]
            X_test = self.X[test_index, :]
            y_test = self.y_binary[test_index, :]
            
            pls = PLSRegression(n_components=n)
            pls.fit(X_train, y_train)

            y_pred = pls.predict(X_test)
            
            # cut off at 0.5 to get only 0 or 1
            y_pred = (y_pred > 0.5).astype("uint8")
            
            # reformat to vector so sklearn accuracy_score works
            y_pred = y_pred.flatten()
            y_test = y_test.flatten()
            
            ac = accuracy_score(y_test, y_pred)
            accuracy.append(ac)
            
            pre = precision_score(y_test, y_pred)
            precision.append(pre)
            
            re = recall_score(y_test, y_pred)
            recall.append(re)

        accuracy = round(np.array(accuracy).mean(), 2)
        precision = round(np.array(precision).mean(), 2)
        recall = round(np.array(recall).mean(), 2)
        
        return (accuracy, precision, recall)

    def _optimize_plsda(self):
        """Optimizes number of components"""
        component = np.arange(1, self.n_components + 1)
        accuracy = []
        for i in component:
            ac, _, _ = self._crossval(i)
            accuracy.append(ac)
            
        best_ac = np.argmax(accuracy)
        return component[best_ac], accuracy
    
    
    def _fit(self, n_comps):
        self._pls = PLSRegression(n_comps)
        self._pls.fit(self.X, self.y_binary)

        self.x_scores = self._pls.x_scores_
        self.y_scores = self._pls.y_scores_
        self.x_weights = self._pls.x_weights_
        self.y_weights = self._pls.y_weights_
        self.y_loadings = self._pls.y_loadings_

        if self.scaling_method is None:
            self.x_loadings = self._pls.x_loadings_
        else:
            self.x_loadings = self._pls.x_loadings_ / self.weights[:, None]
    
    def _get_top_coef_indices(self, n):
        indices = []
        for i in range(len(self.groups)):
            numbers = self._pls.coef_[:, i]
            idx = np.argpartition(numbers, -n)[-n:]
            index = idx[np.argsort((-numbers)[idx])]
            indices.append(index)

        indices = np.sort(np.array(indices).flatten())
        return np.unique(indices)
            
    def calc_vip_scores(self, top_n_coeff=None):
        """
        Calculates variable importance in projection (VIP) scores.
        Optionally only of features with high coefficients.

        Parameters
        ----------
        top_n_coeff : int, optional
            Number highest coefficients per group
            if None calculates all, by default None

        Returns
        -------
        numpy.ndarray
        """
        if top_n_coeff is None:
            w = self.x_weights
        else:
            self._indices = self._get_top_coef_indices(top_n_coeff)
            w = self.x_weights[self._indices, :]

        t = self.x_scores
        q = self.y_loadings

        p, h = w.shape
        
        vips = np.zeros((p,))

        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight)/total_s)

        self.vip_scores = vips
        return vips


    def plot(self, x_comp=1, y_comp=2, annotate=False):
        """
        Plots PLS components as scatter plot.

        Parameters
        ----------
        x_comp : int, optional
            Component x axis, by default 1

        y_comp : int, optional
            Component y axis, by default 2

        annotate : bool, optional
            If True adds sample names to markers,
            by default False

        Returns
        -------
        matplotlib.figure.Figure
        """        
        cols = [f"PLS Component {i}" for i in range(1, self.n_components + 1)]
        df = pd.DataFrame(self.x_scores, columns=cols)
        df["Group"] = self.dataset.labels
        df["Sample"] = self.dataset.samples
        
        with plt.style.context("seaborn"):
            fig = plt.figure()
            sns.scatterplot(
                x=f"PLS Component {x_comp}",
                y=f"PLS Component {y_comp}",
                data=df,
                hue="Group",
                style="Group",
                s=50
                )
            plt.legend(frameon=True, fancybox=True, facecolor="white")
            
            if annotate:
                for _, row in df.iterrows():
                    plt.annotate(
                        row["Sample"],
                        xy=(row[f"PLS Component {x_comp}"], row[f"PLS Component {y_comp}"]),
                        xycoords="data"
                    )

        return fig

    def plot_optimisation(self):
        """
        Plots accuracy from crossvalidation vs number of components
        to find the best parameter.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If optimize is set to False during initial Method.
        """
        if not self.optimize:
            raise ValueError("Can only plot optimization results if optimize argument is True.")
            
        component = np.arange(1, self.n_components + 1)
        best_ac = np.argmax(self._accuracies)
        
        with plt.style.context("seaborn"):
            fig = plt.figure()
            plt.plot(component, self._accuracies)
            plt.scatter(component, self._accuracies)
            plt.plot(
                component[best_ac],
                self._accuracies[best_ac],
                color="tab:orange",
                marker="*",
                markersize=20
                )
            plt.xlabel("Number of PLS Components")
            plt.ylabel("Accuracy")
            plt.title("PLS-DA")
        return fig

    def plot_coefficients(self, group=0):
        """
        Plots PLS coefficients of selected group as image
        with retention and drift time axis.

        Parameters
        ----------
        group : int or str, optional
            Index or name of group, by default 0

        Returns
        -------
        matplotlib.figure.Figure
        """

        if isinstance(group, str):
            group_index = self.groups.index(group)
            group_name = group

        if isinstance(group, int):
            group_index = group
            group_name = self.groups[group]

        coef = self._pls.coef_[:, group_index].\
            reshape(self.dataset[0].values.shape)

        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time
        
        fig, ax = plt.subplots(figsize=(9, 10))
        
        plt.imshow(
            coef,
            cmap="RdBu_r",
            origin="lower",
            aspect="auto",
            extent=(min(drift_time), max(drift_time),
                    min(ret_time), max(ret_time))
            )

        plt.colorbar(label="Coefficients")

        plt.title(f"PLS-DA coefficients of {group_name}", fontsize=14)

        plt.xlabel(self.dataset[0]._drift_time_label, fontsize=12)
        plt.ylabel("Retention Time [s]", fontsize=12)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        return fig

    def plot_vip_scores(self):
        """
        Plots VIP scores as image with retention and drift time axis.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If VIP scores are not calculated.
        """        
        if not hasattr(self.vip_scores):
            raise ValueError("Must calculate VIP scores first.")
        
        vip_matrix = np.zeros(self.X.shape[1])
        vip_matrix[self._indices] = self.vip_scores
        vip_matrix = vip_matrix.reshape(self.dataset[0].values.shape)

        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        fig, ax = plt.subplots(figsize=(9, 10))

        plt.imshow(
            vip_matrix,
            cmap="RdBu_r",
            origin="lower",
            aspect="auto",
            extent=(min(drift_time), max(drift_time),
                    min(ret_time), max(ret_time))
            )

        plt.colorbar(label="VIP scores")

        plt.title(f"PLS-DA VIP scores", fontsize=14)

        plt.xlabel(self.dataset[0]._drift_time_label, fontsize=12)
        plt.ylabel("Retention Time [s]", fontsize=12)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        return fig