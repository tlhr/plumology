'''som - Self-organising-map'''

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


class SOM:
    '''
    SOM - Self-Organising-Map.

    A 2D neural network that clusters high-dimensional data iteratively.

    Parameters
    ----------
    nx : Number of neurons on x-axis.
    ny : Number of neurons on y-axis.
    ndims : Dimension of input data.
    iterations : Total number of iterations to perform.
        Should be at least 10 times the number of neurons.
    learning_rate : The learning rate specifies the
        tradeoff between speed and accuracy of the SOM.
    distance : { 'euclidean', 'periodic' }
        The distance metric to use.
    init : { 'random', 'pca' }
        Initialization method. "pca" uses a grid spanned by the first two
        eigenvectors of the principal component analysis of the input data.
    grid : { 'rect', 'hex' }
        Layout of the SOM, can be either rectangular or hexagonal with
        equidistant nodes. The latter can provide smoother visualization.
    train : { 'seq', 'batch' }
        Training algorithm to use. Sequential picks random feature vectors one
        at a time, while batch mode trains using all features per iteration.
        This can significantly speed up convergence.
    neighbour : { 'gaussian', 'bubble', 'epanechnikov' }
        Type of neighbourhood decay function to use. "bubble" uses a hard
        cutoff, "gaussian" falls off smoothly, and "epanechnikov" starts
        smoothly and ends with a hard cutoff.
    learning : { 'exponential', 'linear' }
        Type of decay for the learning rate. A linear decay can
        improve results in certain cases.
    seed : Seed for the random number generator.

    Attributes
    ----------
    grid : Grid with all x, y positiions of the nodes.
        Useful for visualization.
    weights : Weight vectors of the SOM in shape = (nx, ny, ndims).

    Examples
    --------
    Here we train a 20 by 30 SOM on some colors:

    >>> from plumology.learn import SOM
    >>> som = SOM(20, 30, 3, iterations=400, learning_rate=0.2)
    >>> colors = np.array(
    ...     [[0., 0., 0.],
    ...      [0., 0., 1.],
    ...      [0., 1., 1.],
    ...      [1., 0., 1.],
    ...      [1., 1., 0.],
    ...      [1., 1., 1.],
    ...      [.33, .33, .33],
    ...      [.5, .5, .5],
    ...      [.66, .66, .66]]
    ... )
    >>> som.fit(colors)

    References
    ----------
    Kohonen, T., "Self-Organized Formation of Topologically Correct
    Feature Maps". In: Biological Cybernetics 43 (1): 59–69 (1982).

    '''
    def __init__(
            self,
            nx: int,
            ny: int,
            ndims: int,
            iterations: int,
            learning_rate: float=0.5,
            distance: str='euclid',
            init: str='random',
            grid: str='rect',
            train: str='seq',
            neighbour: str='gaussian',
            learning: str='exp',
            seed: Optional[int]=None
    ) -> None:

        self._iterations = iterations
        self._init_learning_rate = learning_rate
        self._learning_rate = self._init_learning_rate
        self._ndims = ndims
        self._map_radius = max(nx, ny) / 2
        self._dlambda = self._iterations / np.log(self._map_radius)
        self._shape = (nx, ny)
        self._trained = False

        if seed is not None:
            np.random.seed(seed)

        # Establish training algorithm
        if train.startswith('seq'):
            self._type = 's'
        elif train.startswith('batch'):
            self._type = 'b'
        else:
            e = 'Invalid training type! Valid types: sequential, batch'
            raise ValueError(e)

        # Init distance type
        if distance.startswith('euclid'):
            self._dist = self._euclid_dist
        elif distance.startswith('per'):
            self._dist = self._periodic_dist
        else:
            e = 'Invalid distance type! Valid types: euclidean, periodic'
            raise ValueError(e)

        # Init weights
        if init.startswith('r'):
            self.weights = np.random.rand(nx, ny, ndims)
        elif not init.startswith('p'):
            e = 'Invalid initialization type! Valid types: random, pca'
            raise ValueError(e)

        # Init grid
        self._X, self._Y = np.meshgrid(np.arange(ny), np.arange(nx))
        if grid.startswith('r'):
            self._locX = self._X
            self._locY = self._Y
        elif grid.startswith('h'):
            self._locX = np.asarray([
                x + 0.5 if i % 2 == 0 else x
                for i, x in enumerate(self._X.astype(float))
            ])
            self._locY = self._Y * 0.33333
        else:
            e = 'Invalid grid type! Valid types: rect, hex'
            raise ValueError(e)

        # Init neighbourhood function
        if neighbour.startswith('gauss'):
            self._nb = self._nb_gaussian
        elif neighbour.startswith('bub'):
            self._nb = self._nb_bubble
        elif neighbour.startswith('epa'):
            self._nb = self._nb_epanechnikov
        else:
            e = ('Invalid neighbourhood function!' +
                 'Valid types: gaussian, bubble, epanechnikov')
            raise ValueError(e)

        # Init learning-rate function
        if learning.startswith('exp'):
            self._lr = self._lr_exp
        elif learning.startswith('pow'):
            self._final_lr = self._init_learning_rate * np.exp(-1)
            self._lr = self._lr_pow
        elif learning.startswith('lin'):
            self._lr = self._lr_lin
        else:
            e = ('Invalid learning rate function!' +
                 'Valid types: exp, power, linear')
            raise ValueError(e)

        # Output grid for easier plotting
        self.grid = np.asarray(list(zip(self._locX.flatten(),
                                        self._locY.flatten())))

    def _init_weights(self, X: np.ndarray) -> None:
        '''Initialize weights from PCA eigenvectors'''
        if not hasattr(self, 'weights'):
            pca = PCA(n_components=self._ndims)
            comp = pca.fit(X).components_[:2]
            coeff = X.mean(0) + 5 * X.std(0) / self._shape[0]

            # Create grid based on PCA eigenvectors and std dev of features
            raw_weights = np.asarray([
                (coeff * (comp[0] * (x - 0.5 / self._shape[0]) +
                          comp[1] * (y - 0.5 / self._shape[1])))
                for x, y in zip(np.nditer(self._X.flatten()),
                                np.nditer(self._Y.flatten()))
            ]).reshape(self._shape + (self._ndims,))

            # Scale to (0, 1)
            full_shape = self._shape + (1,)
            self.weights = (
                (raw_weights - raw_weights.min(2).reshape(full_shape)) /
                raw_weights.ptp(2).reshape(full_shape)
            )

    def _nb_gaussian(self, dist: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-dist ** 2 / (2 * sigma ** 2))

    def _nb_bubble(self, dist: np.ndarray, sigma: float) -> np.ndarray:
        return dist

    def _nb_epanechnikov(self, dist: np.ndarray, sigma: float) -> np.ndarray:
        return np.maximum(np.zeros_like(dist), 1 - dist ** 2)

    def _lr_exp(self, t: int) -> float:
        return self._init_learning_rate * np.exp(-t / self._iterations)

    def _lr_pow(self, t: int) -> float:
        return (self._init_learning_rate *
                (self._final_lr / self._init_learning_rate) **
                (t / self._iterations))

    def _lr_lin(self, t: int) -> float:
        return (self._init_learning_rate -
                (self._init_learning_rate * t * (np.exp(1) - 1) /
                 (self._iterations * np.exp(1))))

    def _euclid_dist(
            self,
            xmat: np.ndarray,
            index: Tuple[int, int]=(),
            axis: int=2
    ) -> np.ndarray:
        return np.sqrt(((xmat - self.weights[index]) ** 2).sum(axis=axis))

    def _periodic_dist(
            self,
            xmat: np.ndarray,
            index: Tuple[int, int]=(),
            axis: int=2
    ) -> np.ndarray:
        pi2 = np.pi * 2
        dx = (xmat - self.weights[index]) / pi2
        return np.sqrt((((dx - round(dx)) * pi2) ** 2).sum(axis=axis))

    def _train(self, X: np.ndarray) -> None:
        for t in range(self._iterations):
            # Update learning rate, reduce radius
            lr = self._lr(t)
            neigh_radius = self._map_radius * np.exp(-t / self._dlambda)

            # Choose random feature vector
            f = X[np.random.choice(len(X))]

            # Calc euclidean distance
            xmat = np.broadcast_to(f, self._shape + (self._ndims,))
            index = self._dist(xmat).argmin()
            bmu = np.unravel_index(index, self._shape)

            # Create distance matrix
            distmat = (
                (self._locX - self._locX[bmu]) ** 2 +
                (self._locY - self._locY[bmu]) ** 2
            ).reshape(self._shape + (1,))

            # Mask out unaffected nodes
            mask = (distmat < neigh_radius).astype(int)
            theta = self._nb(distmat * mask, neigh_radius)
            self.weights += mask * theta * lr * (f - self.weights)

    def _batch_train(self, X: np.ndarray) -> None:
        for t in range(self._iterations):
            # Update learning rate, reduce radius
            lr = self._lr(t)
            neigh_radius = self._map_radius * np.exp(-t / self._dlambda)

            for f in X:
                # Calc euclidean distance
                xmat = np.broadcast_to(f, self._shape + (self._ndims,))
                index = self._dist(xmat).argmin()
                bmu = np.unravel_index(index, self._shape)

                # Create distance matrix
                distmat = (
                    (self._locX - self._locX[bmu]) ** 2 +
                    (self._locY - self._locY[bmu]) ** 2
                ).reshape(self._shape + (1,))

                # Mask out unaffected nodes
                mask = (distmat < neigh_radius).astype(int)
                theta = self._nb(distmat * mask, neigh_radius)
                self.weights += mask * theta * lr * (f - self.weights)

    def fit(self, X: np.ndarray) -> None:
        '''
        Run the SOM.

        Parameters
        ----------
        X : input data as array of vectors.

        '''
        self._init_weights(X)
        if self._type == 's':
            self._train(X)
        else:
            self._batch_train(X)
        self._trained = True

    def create_index(self, X: np.ndarray) -> None:
        '''
        Create an index grid, allowing the coloring of the map with arbitrary
        feature data. For instance, one could train the SOM on a subset of the
        data, and then create an index using the full dataset. The transform()
        method will only need to check the created index grid featuring the
        best matching datapoint index per node.

        Parameters
        ----------
        X : input data as used to train the SOM, can be significantly larger.

        '''
        if not self._trained:
            raise ValueError('You need to train the SOM first!')

        self.index = np.zeros(self._shape, dtype=np.int32)

        # For each node we calculate the distance to each datapoint
        for index in np.ndindex(self._shape):
            self.index[index] = self._dist(X, index=index, axis=1).argmin()

    def transform(self, X: np.ndarray) -> np.ndarray:
        '''
        Transform a dataset based on the index grid created by index().
        This method will return a subset of the dataset in the shape of
        the node matrix.

        Parameters
        ----------
        X : input data

        Returns
        -------
        grid : subset of the input data assigned to the best nodes

        '''
        if not self._trained:
            raise ValueError('You need to train the SOM first!')
        if not hasattr(self, 'index'):
            raise ValueError('You need to index the SOM first!')

        grid = np.zeros(self._shape)
        for index in np.ndindex(self.index.shape):
            grid[index] = X[self.index[index]]

        return grid
