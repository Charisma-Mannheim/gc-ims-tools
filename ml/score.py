from ims_module.ml import BaseModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier


# TODO: Add similar features for regression


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
            GCIMS_DataSet or Spectra

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

    # TODO: rewrite using plotly
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
    
# TODO: Implement ROC_AUC class with plots for models
# class ROC_AUC(BaseModel):
    
#     def __init__(self, dataset, model, scaling_method):
#         super().__init__(dataset, scaling_method)
#         self.model = model
        
        
#     def plot_roc(self):
#         pass


class CompareClassifier(BaseModel):

    def __init__(self, dataset, append_models=None, models=None,
                 test_size=0.2, kfold=5, scaling_method=None):
        """
        Compares classification algorithms by performing
        a cross validation on training data and predicting
        labels for the test data.

        Can use anything that follows the scikit learn API.
        Contains a default list of classifiers.

        Uses joblib with "loky" backend to parallelize calculations.

        Parameters
        ----------
        dataset : varies
            Can be any dataset class with an get_xy method
            like GCIMS_DataSet or Spectra

        append_models : sklearn model, optional
            Appends model to default list,
            by default None

        models : list, optional
            Replaces default list, by default None

        test_size : float, optional
            For train test split, by default 0.2

        kfold : int, optional
            For cross validation, by default 5

        scaling_method : str, optional
            'standard', 'auto', 'pareto' and 'var' are valid arguments,
            by default None
        """
        super().__init__(dataset, scaling_method, test_size)
        self.append_models = append_models
        self.models = models
        self.kfold = kfold

        with joblib.parallel_backend('loky'):
            if self.models is None:
                self.models = self._get_default_model_list()
            if self.append_models is not None:
                self.models.append(self.append_models)

            self.result = self._fit()

    @staticmethod
    def _get_default_model_list():
        return [
                GradientBoostingClassifier(),
                XGBClassifier(),
                RandomForestClassifier(),
                AdaBoostClassifier(),
                SVC(),
                GaussianNB(),
                KNeighborsClassifier(),
                GaussianProcessClassifier(),
                DecisionTreeClassifier()
            ]

    def _fit(self):
        names = []
        scores = []
        accuracy = []
        for model in self.models:
            names.append(str(model).strip('()'))
            score = cross_val_score(
                model, self.X_train,
                self.y_train, cv=self.kfold
                )
            scores.append(score.mean())
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy.append(accuracy_score(self.y_test, y_pred))

        result = pd.DataFrame(
            {'Model': names,
             'CV 10-fold': scores,
             'Accuracy Test Data': accuracy}
            )
        result = result.sort_values('Accuracy Test Data', ascending=False)
        return result


class TPOT_Optimizer(BaseModel):

    def __init__(self, dataset, scaling_method=None, 
                 test_size=0.2, generations=5,
                 population_size=100, kfold=5, verbosity=2,
                 config_dict=None, use_dask=False):
        """
        Automatically optimizes scikit-learn pipeline
        using TPOT.
        
        http://epistasislab.github.io/tpot/

        Parameters
        ----------
        dataset : varies
            Can be any dataset class with an get_xy method
            like GCIMS_DataSet or Spectra

        scaling_method : str, optional
            'standard', 'auto', 'pareto' and 'var' are valid arguments,
            by default None

        test_size : float, optional
            For train test split,
            by default 0.2

        generations : int, optional
            Number of iterations to the run pipeline optimization process.
            It must be a positive number or None. If None,
            the parameter max_time_mins must be defined as the runtime limit.
            Generally, TPOT will work better when you give it more generations
            (and therefore time) to optimize the pipeline.
            TPOT will evaluate population_size + generations × offspring_size
            pipelines in total,
            by default 5

        population_size : int, optional
            Number of individuals to retain in the genetic
            programming population every generation. Must be a positive number.
            Generally, TPOT will work better when you give it
            more individuals with which to optimize the pipeline,
            by default 100

        kfold : int, optional
            For cross validation, by default 5

        verbosity : int, optional
            How much information TPOT communicates while it's running.
            
            Possible inputs are:
            0, TPOT will print nothing,
            1, TPOT will print minimal information,
            2, TPOT will print more information and provide a progress bar,
            3, TPOT will print everything and provide a progress bar,
            by default 2

        config_dict : str, optional
            A configuration dictionary for customizing the operators
            and parameters that TPOT searches in the optimization process.
            
            Possible inputs are:
            Python dictionary, TPOT will use your custom configuration,
            
            string 'TPOT light', TPOT will use a built-in configuration
                with only fast models and preprocessors,
                
            string 'TPOT MDR', TPOT will use a built-in configuration specialized for genomic studies,
            
            string 'TPOT sparse': TPOT will use a configuration dictionary
            with a one-hot encoder and the operators normally included
            in TPOT that also support sparse matrices,
            
            None, TPOT will use the default TPOTClassifier configuration,
            by default None

        use_dask : bool, optional
            Whether to use Dask-ML's pipeline optimiziations.
            This avoid re-fitting the same estimator
            on the same split of data multiple times.
            It will also provide more detailed diagnostics
            when using Dask's distributed scheduler.
            (XGBoost and dask are not compatible on windows),
            by default False
        """        
        super().__init__(dataset, scaling_method, test_size)
        self.generations = generations
        self.population_size = population_size
        self.kfold = kfold
        self.verbosity = verbosity
        self.config_dict = config_dict
        self.use_dask = use_dask

        self.tpot = TPOTClassifier(
            generations=self.generations,
            population_size=self.population_size,
            cv=self.kfold,
            n_jobs=-1,
            random_state=0,
            verbosity=self.verbosity,
            config_dict=self.config_dict,
            use_dask=self.use_dask,
            memory='auto'
        )
        self.tpot.fit(self.X_train, self.y_train)
        self.score = self.tpot.score(self.X_test, self.y_test)
        self.best_pipeline = self.tpot.fitted_pipeline_

    def export_model(self, name='best_pipeline.py'):
        """
        Exports best pipeline as python script.

        Parameters
        ----------
        name : str, optional
            file name, by default 'best_pipeline.py'
        """
        self.tpot.export(name)
