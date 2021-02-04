import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from natsort import natsorted


class BaseModel:

    def __init__(self, dataset, scaling_method=None, test_size=0.2):
        """
        Baseclass for all machine learning related classes.
        Interface between datasets and algorythms.
        Handels formatting, scaling and splitting of data.

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
        """
        self.dataset = dataset
        self.X, self.y = self.dataset.get_xy()

        self.scaling_method = scaling_method
        if scaling_method == 'standard':
            self.X = StandardScaler().fit_transform(self.X)
        if scaling_method is not None and scaling_method != 'standard':
            self.weights = self._calc_weights()
            self.X = self.X * self.weights

        self.test_size = test_size
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


def load_npy(path, flatten=False):
    """
    Loads all npy files produced by any of the
    to_npy methods in directory.

    Parameters
    ----------
    path : str
        Directory with data.

    flatten : bool, optional
        Flattens data to 2D, by default False

    Returns
    -------
    tuple
        (features, labels)
    """
    labels = np.load(f'{path}/labels.npy')
    data_paths = natsorted(glob(f'{path}/data/*npy'))
    data = []
    for i in data_paths:
        data.append(np.load(i))
    if flatten:
        data = [i.flatten() for i in data]
    else:
        data
    data = np.stack(data)
    return (data, labels)

# def iterate_npy(path, flatten=False):
#     labels = np.load(f'{path}/labels.npy')
#     data_paths = natsorted(glob(f'{path}/data/*npy'))
#     for i, j in zip(data_paths, labels):
#         if flatten:
#             yield (np.load(i).flatten(), j)
#         else:
#             yield (np.load(i), j)
