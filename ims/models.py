import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report


class BaseModel:
    
    def __init__(self, dataset, scaling_method=None, test_size=None):
        self.dataset = dataset
        self.X, self.y = self.dataset.get_xy()
        
        self.scaling_method = scaling_method
        if scaling_method == 'standard':
            self.X = StandardScaler().fit_transform(self.X)
        if scaling_method is not None and scaling_method != 'standard':
            self.weights = self._calc_weights()
            self.X = self.X * self.weights

        self.test_size = test_size
        if self.test_size is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size
            )

    def _calc_weights(self):
        '''
        Calculate weights for scaling dependent on the method.
        '''
        if self.scaling_method == 'auto':
            weights = 1/np.std(self.X, 0)
        elif self.scaling_method == 'pareto':
            weights = 1/np.sqrt(np.std(self.X, 0))
        elif self.scaling_method == 'var':
            weights = 1/np.var(self.X, 0)
        else:
            raise ValueError(f'{self.scaling_method} is not a supported method')
        weights = np.nan_to_num(weights, posinf=0, neginf=0)
        return weights
    

class PCA_Model(BaseModel):
    
    def __init__(self, dataset, scaling_method=None, **kwargs):
        """
        Wrapper class for scikit learn PCA.
        Adds plots for explained variance ratio,
        loadings and scatter plots of components.

        Parameters
        ----------
        dataset : varies
            Dataset or Spectra

        scaling_method : str, optional
            'standard', 'auto', 'pareto' and 'var' are valid arguments,
            by default None
        """
        super().__init__(dataset, scaling_method)
        self.pca = PCA(**kwargs).fit(self.X)
        self.pc = self.pca.transform(self.X)
        if self.scaling_method is not None and scaling_method != 'standard':
            self.loadings = self.pca.components_ / self.weights
        else:
            self.loadings = self.pca.components_

    def __repr__(self):
        return f'''PCA:
{self.dataset.name},
{self.scaling_method} scaling
'''

    def scatter_plot(self, PC_x=1, PC_y=2):
        """
        scatter_plot
        Scatter plot of two principal components.

        Parameters
        ----------
        PCs : tuple, optional
            The principal components to plot, by default (1, 2)

        Returns
        -------
        matplotlib.pyplot.figure
        """        
        pc_df = pd.DataFrame(
            data=self.pc,
            columns=[f"PC {x}" for x in range(1, self.pca.n_components_ + 1)]
        )
        pc_df['Sample'] = self.dataset.samples
        pc_df['Label'] = self.dataset.labels

        with plt.style.context("seaborn"):
            fig = sns.scatterplot(
                x=f"PC {PC_x}",
                y=f"PC {PC_y}",
                data=pc_df,
                hue="Label",
                markers="Label"
                )
            plt.legend(frameon=True, fancybox=True, facecolor="white")

        return fig

    def loadings_plot(self, PC=1, color_range=0.1, width=9, height=10):
        """
        Plots loadings of a principle component with the original retention
        and drift time coordinates.

        Parameters
        ----------
        PC : int, optional
            principal component, by default 1

        color_range : int, optional
            color_scale ranges from - color_range to + color_range
            centered at 0

        width : int, optional
            plot width in inches, by default 9

        height : int, optional
            plot height in inches, by default 10

        Returns
        -------
        matplotlib.pyplot.figure
        """        
        # use retention and drift time axis from the first spectrum
        ret_time = self.dataset[0].ret_time
        drift_time = self.dataset[0].drift_time

        loading_pc = self.loadings[PC-1, :].reshape(len(ret_time), len(drift_time))

        fig, ax = plt.subplots(figsize=(width, height))

        plt.imshow(
            loading_pc,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=(-color_range),
            vmax=color_range
            )

        plt.colorbar()

        xlocs, _ = plt.xticks()
        ylocs, _ = plt.yticks()

        rt_ticks = [round(ret_time[int(i)]) for i in ylocs[1:-1]]
        dt_ticks = [round(drift_time[int(i)], 1) for i in xlocs[1:-1]]

        plt.xticks(xlocs[1:-1], dt_ticks)
        plt.yticks(ylocs[1:-1], rt_ticks)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.xlabel(self.dataset[0]._drift_time_label, fontsize=12)
        plt.ylabel("Retention Time [s]", fontsize=12)
        plt.title(f"PCA Loadings of PC {PC}", fontsize=16)
        return fig

    def expl_var_ratio_plot(self):
        """
        expl_var_ratio_plot
        Plots the explained variance ratio per principal component
        and cumulatively.

        Returns
        -------
        matplotlib.pyplot.figure
        """        
        x = [*range(1, self.pca.n_components_ + 1)]
        y = self.pca.explained_variance_ratio_

        with plt.style.context("seaborn"):
            fig, ax = plt.subplots()
            
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(MaxNLocator(integer=True))
            
            plt.xticks(x)
            plt.xlabel("Principal Component")
            plt.ylabel("Explainded variance ratio [%]")
            
            ax.plot(x, np.cumsum(y) * 100, label="cumulative")
            ax.plot(x, y * 100, label="per PC")
            
            plt.legend(frameon=True, fancybox=True, facecolor="white")

        return fig


class CrossVal(BaseModel):

    def __init__(self, dataset, model,
                 scaling_method=None, kfold=10):
        """
        Performs cross validation and presents results.
        Can be used with all scikit learn models
        or anything that follows the same API like sklearn pipelines
        XGBoostClassifier, or keras sklearn wrapper ANNs.

        Parameters
        ----------
        dataset : varies
            Dataset or Spectra

        model : scikit learn model
            Or anything that follows the scikit learn API.

        scaling_method : str, optional
            'standard', 'auto', 'pareto' and 'var' are valid arguments,
            by default None

        kfold : int, optional
            Number of splits, by default 10
        """
        super().__init__(dataset, scaling_method)
        self.model = model
        self.kfold = kfold

        with joblib.parallel_backend('loky'):
            self.scores = cross_val_score(
                self.model, self.X, self.y,
                cv=self.kfold
                )
            self.predictions = cross_val_predict(
                self.model, self.X,
                self.y, cv=self.kfold
                )

        self.score = self.scores.mean()
        self.confusion_matrix = confusion_matrix(self.y, self.predictions)
        self.report =  pd.DataFrame(
            classification_report(
                self.y, self.predictions, output_dict=True
            )
        )
        self.result = pd. DataFrame(
            {'Sample': self.dataset.samples,
             'Actual': self.y,
             'Predicted': self.predictions}
        )
        self.mismatch = self.result[self.result.Actual != self.result.Predicted]

    def __repr__(self):
        return f'''
CrossVal:
{self.dataset.name},
Scaling: {self.scaling_method},
{self.model}, {self.kfold} fold
'''


    def plot_confusion_matrix(self):
        """
        Plots confusion matrix.

        Returns
        -------
        matplotlib axis object
            Uses seaborn.
        """
        cm_df = pd.DataFrame(self.confusion_matrix,
                             index=np.unique(self.y),
                             columns=np.unique(self.y))

        sns.set(rc={'figure.figsize':(8, 6)})
        sns.set_style('ticks')
        sns.set_context('talk')
        ax = sns.heatmap(cm_df, cbar=False, annot=True, cmap='Reds')
        ax.set(xlabel='Predicted', ylabel='Actual')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        return ax
    