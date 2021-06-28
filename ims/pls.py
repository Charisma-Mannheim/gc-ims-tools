import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    accuracy_score
    )
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
    
    def __init__(self, dataset, scaling_method=None, optimize=True,
                 n_components=20, kfold=10):
        super().__init__(dataset, scaling_method)
        self.optimize = optimize
        self.n_components = n_components
        self.kfold = kfold

        self.groups = np.unique(self.dataset.labels)
        self.y_binary = np.zeros((len(self.dataset), len(self.groups)))
        for i, j in enumerate(self.groups):
            col = [j in label for label in self.dataset.labels]
            self.y_binary[:, i] = col

        if self.optimize:
            self._best_comp = self._optimise_pls()

        self._fit()
        # self.vip_scores = self._calc_vips()
    
    
    def _predict(self, X_train, y_train, X_test):
        plsda = PLSRegression(n_components=2)
        plsda.fit(X_train, y_train)
        binary_prediction = (pls_binary.predict(X_test)[:,0] > 0.5).astype('uint8')
        return binary_prediction

    def _crossval(self,X_train, y_train, y_test, kfold):

        accuracy = []
        cval = KFold(n_splits=10, shuffle=True, random_state=19)
        for train, test in cval.split(X_binary):
            
            y_pred = self._predict(X_binary[train,:], y_binary[train], X_binary[test,:])
            
            accuracy.append(accuracy_score(y_binary[test], y_pred))

        return np.array(accuracy).mean()
            
    
    def _optimise_pls(self):
        accuracy = []
        component = np.arange(1, self.n_components)
        
        for i in component:
            pls = PLSRegression(n_components=i)
            y_cv = cross_val_predict(pls, self.X, self.y_binary,
                                     cv=self.kfold)
            accuracy.append(accuracy_score(self.y, y_cv))
            
        best_ac = np.argmin(accuracy)
        


        return component[best_ac]
    
    def plot_optimisation(self):
        with plt.style.context("seaborn"):
            plt.plot(component, accuracy)
            plt.scatter(component, accuracy)
            plt.plot(component[best_ac], accuracy[best_ac], color="tab:orange",
                        marker="*", markersize=20)
            plt.xlabel("Number of PLS Components")
            plt.ylabel("MSE")
            plt.title("PLS")
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
        
    def _calc_vips(self):
        """https://github.com/scikit-learn/scikit-learn/issues/7050"""
        t = self.x_scores
        w = self.x_weights
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
    
    def plot_coef(self, group=1):
        """Plots coefficients"""
        coef = self._pls.coef_[:, group-1].\
            reshape(self.dataset[0].values.shape)
        
        fig, ax = plt.subplots(figsize=(9, 10))
        plt.imshow(coef, cmap="RdBu_r", origin="lower", aspect="auto")
        plt.colorbar()

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
    
    def plot_vip(self):
        pass