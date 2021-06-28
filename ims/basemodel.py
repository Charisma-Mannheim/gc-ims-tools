import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
            weights = 1 / np.std(self.X, 0)
        elif self.scaling_method == 'pareto':
            weights = 1 / np.sqrt(np.std(self.X, 0))
        elif self.scaling_method == 'var':
            weights = 1 / np.var(self.X, 0)
        else:
            raise ValueError(f'{self.scaling_method} is not a supported method')
        weights = np.nan_to_num(weights, posinf=0, neginf=0)
        return weights
