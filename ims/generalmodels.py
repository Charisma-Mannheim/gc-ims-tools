import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    accuracy_score
    )
from ims import BaseModel


class Classification(BaseModel):

    def __init__(self, dataset, model, scaling_method=None,
                 validation_method="cross validation", kfold=10):
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
            
        validation_method : str, optional
            'cross validation', 'leave one out' and 'bootstrapping' are valid arguments,
            by default 'cross validation'

        kfold : int, optional
            Number of splits for cross validation
            ignored for 'leave one out' and 'bootstrapping',
            by default 10
        """
        super().__init__(dataset, scaling_method)
        self.model = model
        self.kfold = kfold
        self.validation_method = validation_method
        if self.validation_method == "cross validation" or "leave one out":
            self._crossval()
        elif self.validation_method == "bootstrapping":
            self._bootstrap()

    def _crossval(self):
        """Performs crossvalidation with either kfold split or leave one out"""

        if self.validation_method == "cross validation":
            cv = self.kfold
        elif self.validation_method == "leave one out":
            cv = LeaveOneOut()
        
        self.scores = cross_val_score(
            self.model, self.X, self.y,
            cv=cv
            )
        self.predictions = cross_val_predict(
            self.model, self.X,
            self.y, cv=cv
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

    def _bootsrap(self):
        pass

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

        sns.set(rc={'figure.figsize':(7, 6)})
        ax = sns.heatmap(cm_df, cbar=False, annot=True, cmap='Reds')
        ax.set(xlabel='Predicted', ylabel='Actual')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        return ax
    

class Regression(BaseModel):

    def __init__(self, dataset, model, scaling_method=None,
                 validation_method="cross validation", kfold=10):
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
            
        validation_method : str, optional
            'cross validation' and 'bootstrapping' are valid arguments,
            by default 'cross validation'

        kfold : int, optional
            Number of splits, by default 10
        """
        super().__init__(dataset, scaling_method)
        self.model = model
        self.kfold = kfold
        self.validation_method = validation_method
        if self.validation_method == "cross validation":
            self._crossval()
        elif self.validation_method == "bootstrapping":
            self._bootstrap()

    def _crossval(self):
        self.scores = cross_val_score(
            self.model, self.X, self.y,
            cv=self.kfold
            )
        self.score = self.scores.mean()
        self.prediction = cross_val_predict(
            self.model, self.X,
            self.y, cv=self.kfold
            )
        
        self.r2_score = round(r2_score(self.y, self.prediction), 3)
        self.accuracy = round(mean_squared_error(self.y, self.prediction), 3)

    def _bootstrap(self):
        pass
    
    def _leave_one_out(self):
        pass

    def plot(self):
        z = np.polyfit(self.y, self.prediction, 1)

        fig = plt.figure()
        plt.scatter(self.prediction, self.y)
        plt.plot(self.y, self.y, label="Ideal", c="tab:green", linewidth=1)
        plt.plot(np.polyval(z, self.y), self.y, label="Regression",
                    c="tab:orange", linewidth=1)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        # plt.title("Crossvalidation")
        plt.legend(frameon=True, fancybox=True, facecolor="white")
        return fig