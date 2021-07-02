import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from ims import BaseModel


class PLSR(BaseModel):
    
    def __init__(self, dataset, scaling_method=None, max_comp=20, 
                 kfold=10, plot_components=True, **kwargs):
        
        super().__init__(dataset, scaling_method)
        self.kfold = kfold
        self.max_comp = max_comp
        self.plot_components = plot_components

        self._best_comp = self.optimise_pls()
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

    def optimise_pls(self):
        accuracy = []
        component = np.arange(1, self.max_comp)
        
        for i in component:
            pls = PLSRegression(n_components=i)
            y_cv = cross_val_predict(pls, self.X, self.y, cv=self.kfold)
            accuracy.append(mean_squared_error(self.y, y_cv))
            
        best_ac = np.argmin(accuracy)
        
        if self.plot_components:
            with plt.style.context("seaborn"):
                plt.plot(component, accuracy)
                plt.scatter(component, accuracy)
                plt.plot(component[best_ac], accuracy[best_ac], color="tab:orange",
                         marker="*", markersize=20)
                plt.xlabel("Number of PLS Components")
                plt.ylabel("MSE")
                plt.title("PLS")
                plt.show()

        return component[best_ac]

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
                 n_components=20, kfold=5, n_vips=None):
        super().__init__(dataset, scaling_method)
        self.optimize = optimize
        self.n_components = n_components
        self.kfold = kfold
        self.n_vips = n_vips

        self.groups = list(np.unique(self.dataset.labels))
        self.y_binary = np.zeros((len(self.dataset), len(self.groups)))
        for i, j in enumerate(self.groups):
            col = [j in label for label in self.dataset.labels]
            self.y_binary[:, i] = col

        if self.optimize:
            self._best_comp, self._accuracy = self._optimise_plsda()

        self._fit()
        
        if self.n_vips is not None:
            self._indices = self._get_top_coef_indices()
            self.vip_scores = self._calc_vips()


    def _crossval(self, n):
        '''Crossvalidation zu optimize number of components'''
        kf = KFold(self.kfold, shuffle=True, random_state=1)

        accuracy = []
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

        return np.array(accuracy).mean()

    def _optimise_plsda(self):
        """Optimizes number of components"""
        component = np.arange(1, self.n_components + 1)
        accuracy = []
        for i in component:
            ac = self._crossval(i)
            accuracy.append(ac)
            
        best_ac = np.argmax(accuracy)
        return component[best_ac], accuracy
    
    def plot_optimisation(self):
        component = np.arange(1, self.n_components + 1)
        with plt.style.context("seaborn"):
            plt.plot(component, self._accuracy)
            plt.scatter(component, self._accuracy)
            plt.plot(
                component[self._best_comp],
                self._accuracy[self._best_comp - 1],
                color="tab:orange",
                marker="*",
                markersize=20
                )
            plt.xlabel("Number of PLS Components")
            plt.ylabel("Accuracy in %")
            plt.title("PLS-DA")
            plt.show()
    
    def _fit(self):
        if self.optimize:
            n_comps = self._best_comp
        else:
            n_comps = self.n_components
        
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
    
    def _get_top_coef_indices(self):
        n = self.n_vips
        indices = []
        for i in range(len(self.groups)):
            numbers = self._pls.coef_[:, i]
            idx = np.argpartition(numbers, -n)[-n:]
            index = idx[np.argsort((-numbers)[idx])]
            indices.append(index)

        indices = np.sort(np.array(indices).flatten())
        return np.unique(indices)
            
    def _calc_vips(self):
        """https://github.com/scikit-learn/scikit-learn/issues/7050"""
        
        t = self.x_scores
        w = self.x_weights[self._indices, :]
        q = self.y_loadings
        
        p, h = w.shape
        
        vips = np.zeros((p,))
        
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight)/total_s)

        return vips

    def plot(self, x_comp=1, y_comp=2):
        """Plots PLS components as scatter plot """
        cols = [f"PLS Comp {i}" for i in range(self.n_components)]
        df = pd.DataFrame(self.x_scores, columns=cols)
        df["Group"] = self.dataset.labels
        
        with plt.style.context("seaborn"):
            fig = plt.figure()
            sns.scatterplot(x=f"PLS Comp {x_comp}", y=f"PLS Comp {y_comp}",
                            data=df, hue="Group", style="Group", s=50)
            plt.legend(frameon=True, fancybox=True, facecolor="white")
        return fig
    
    def plot_coef(self, group=0):
        """Plots coefficients"""
        
        if isinstance(group, str):
            group_index = self.groups.index(group)
            group_name = group

        if isinstance(group, int):
            group_index = group
            group_name = self.groups[group]
        
        coef = self._pls.coef_[:, group_index].\
            reshape(self.dataset[0].values.shape)

        fig, ax = plt.subplots(figsize=(9, 10))
        plt.imshow(coef, cmap="RdBu_r", origin="lower", aspect="auto")
        plt.colorbar()

        plt.title(f"PLS-DA coefficients of {group_name}", fontsize=14)

        plt.xlabel(self.dataset[0]._drift_time_label, fontsize=12)
        plt.ylabel("Retention Time [s]", fontsize=12)

        xlocs, _ = plt.xticks()
        ylocs, _ = plt.yticks()

        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        rt_ticks = [round(ret_time[int(i)]) for i in ylocs[1:-1]]
        dt_ticks = [round(drift_time[int(i)], 1) for i in xlocs[1:-1]]

        plt.xticks(xlocs[1:-1], dt_ticks)
        plt.yticks(ylocs[1:-1], rt_ticks)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        return fig

    def plot_vips(self):
        vip_matrix = np.zeros(self.X.shape[1])
        vip_matrix[self._indices] = self.vip_scores
        vip_matrix = vip_matrix.reshape(self.dataset[0].values.shape)

        fig, ax = plt.subplots(figsize=(9, 10))
        plt.imshow(vip_matrix, cmap="RdBu_r", origin="lower", aspect="auto")
        plt.colorbar(label="VIP scores")

        plt.title(f"PLS-DA VIP scores for top {self.n_vips} coefficients",
                  fontsize=14)

        plt.xlabel(self.dataset[0]._drift_time_label, fontsize=12)
        plt.ylabel("Retention Time [s]", fontsize=12)

        xlocs, _ = plt.xticks()
        ylocs, _ = plt.yticks()

        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        rt_ticks = [round(ret_time[int(i)]) for i in ylocs[1:-1]]
        dt_ticks = [round(drift_time[int(i)], 1) for i in xlocs[1:-1]]

        plt.xticks(xlocs[1:-1], dt_ticks)
        plt.yticks(ylocs[1:-1], rt_ticks)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        return fig