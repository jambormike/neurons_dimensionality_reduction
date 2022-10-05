# ADJUST os parameters to avoid oversubscription before numpy is called

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6



import numpy as np

import random

from scipy.stats import zscore
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

import itertools
import multiprocessing as mp

import umap


### Place cell settings

# room dimensions in cm. (might break if changed)
width = 140
height = 140

# number of simulated place cells
n_cells = 75

# number of steps inside the environment
minutes = 5
nSteps = minutes * 600
print(str(minutes), "minutes is", str(nSteps), "time bins of 100ms")

# nSteps = 100000

# covariance matrix sigma for place cell class
sigma = np.array([[80, 0],
                [0, 80]])


# place cell class

class PlaceCell(object):

    def __init__(self, mu, sigma):

        self.mu = mu
        self.sigma = sigma
        
    def gaussian(self, x):

        xm = x - self.mu
        sigma = self.sigma
        
        factor1 = 4            # peak firing rate
        factor2 = (np.exp((-1/2) * xm.T @ np.linalg.inv(sigma) @ xm))
        result = factor1 * factor2
        
        return result
        
    def activity(self, x):
        activity = np.random.poisson(self.gaussian(x))

        return activity


def generatePlaceCells(n_cells, sigma=sigma, height= height, width= width):

    cellsList = []
    for i in range(n_cells):
        # mu being location at which the cell is most active 
        mu = np.array([[random.uniform(0,width)],       # width
                       [random.uniform(0,height)]])     # height
        
        cellsList.append(PlaceCell(mu, sigma))
        
    return cellsList


def randomMouse(cellsList, nSteps, height=height, width= width):

    data = []
    dataLocation = []
    for i in range(nSteps):
        location = np.array([[random.uniform(0,width)],
                             [random.uniform(0,height)]])
        # dataLocation.append(location)
        dataLocation.append(np.copy(location))

        sample = []
        for placeCell in cellsList:
            sample.append(placeCell.activity(location))
        data.append(np.hstack(np.copy(sample)))
        
    data = np.vstack(data)
    dataLocation = np.hstack(dataLocation)
        
    return (data, dataLocation)

def distanceMeasure(coordinates, coordinate_indicies):

    locations = []
    for i in coordinate_indicies:
        locations.append(coordinates[: , i])

    locations_copy = locations.copy()
    used_locations = []
    distances = []

    for i in range(len(locations)):
        current_coordinate = locations_copy.pop(0)
        try:
            distance = np.linalg.norm(current_coordinate - used_locations[-1])
        except IndexError:
            distance = 0

        distances.append(distance)
        used_locations.append(current_coordinate)

    return distances[1:]


def pca_function(randomZscore):

    # from sklearn.decomposition import PCA                                               # I don't have to import it here, unlike UMAP

    randomMousePCA = PCA(n_components= 2)
    randomPCA_scores = randomMousePCA.fit_transform(randomZscore)


    return randomPCA_scores


def umap_function(randomZscore, n_jobs=1):
    
    import umap                                                                         # for some reason I have to import it in the function
    
    randomReducer = umap.UMAP(n_jobs=n_jobs)     # n_neighbors=
    randomUMAP_embedding = randomReducer.fit_transform(randomZscore)

    return randomUMAP_embedding


def laplacian_function(randomZscore, n_jobs=1):

    randomLaplacian = SpectralEmbedding(n_components= 2, n_jobs=n_jobs)
    randomLaplacianEmbedding = randomLaplacian.fit_transform(randomZscore)

    return randomLaplacianEmbedding


def main_loop(
    n_cells: int = 30,
    minutes: int = 30,
    kLoops: int = 6,
    desired_n_samples: int = 5000):


    nSteps = minutes * 600
    cell_List = generatePlaceCells(n_cells)
    (randomMouseData, randomMouseLocation) = randomMouse(cell_List, nSteps)
    randomZscore = np.nan_to_num(zscore(randomMouseData))
    print("data generated, randomZscore(data) shape is " + str(randomZscore.shape) +", randomMouseLocation shape is " + str(randomMouseLocation.shape))

    data_list=[]
    time_cut = 3000
    for k in range(1, kLoops +1):
        print("_" * 100)
        lowest_data_list = []
        print("Inner loop k:", k,"out of", kLoops, "initiated")
        randomMouseLocation_cut = randomMouseLocation[:,:time_cut]
        randomZscore_cut = randomZscore[:time_cut,:]

        pca_embedding = pca_function(randomZscore_cut).T
        print("PCA done,", "pca_embedding shape is", pca_embedding.shape)

        laplacian_embedding = laplacian_function(randomZscore_cut).T
        print("Laplacian done,", "Laplacian_embedding shape is", laplacian_embedding.shape)

        umap_embedding = umap_function(randomZscore_cut).T
        print("UMAP done,", "UMAP_embedding shape is", umap_embedding.shape)

        if time_cut < desired_n_samples:
            n_samples = time_cut
        else:
            n_samples = desired_n_samples

        coordinate_indicies = random.sample(range(time_cut), n_samples)
        real_distance = distanceMeasure(randomMouseLocation_cut, coordinate_indicies)
        pca_distance = distanceMeasure(pca_embedding, coordinate_indicies)
        laplacian_distance = distanceMeasure(laplacian_embedding, coordinate_indicies)
        umap_distance = distanceMeasure(umap_embedding, coordinate_indicies)

        real_data = [randomMouseLocation_cut, real_distance]
        lowest_data_list.append(real_data)
        pca_data = [pca_embedding, pca_distance]
        lowest_data_list.append(pca_data)
        laplacian_data = [laplacian_embedding, laplacian_distance]
        lowest_data_list.append(laplacian_data)
        umap_data = [umap_embedding, umap_distance]
        lowest_data_list.append(umap_data)

        data_list.append(lowest_data_list)
        time_cut += 3000

    return data_list, cell_List

def main_loop_wrapper(args):
    return main_loop(*args)


if __name__ == "__main__":

     ### Main loop settings

    n_cells = 10                        # number of cells to begin with
    cells_addition = 10                 # number of cells to add per i loop

    desired_n_samples = 5000            # number of locations to compare distances between

    iLoops = 30                         # number of loops for the main loop i
    jLoops = 5                          # number of experiments
    kLoops = 6                          # number of time additions (+5min, max 6)

    minutes = 30                        # time of the longest experiment

    master_data_list = []               # list to contain all axperimental results
    master_cell_list = []               # list to contain all cell lists

    ####
    n_cpu = 64

    # Construct a vector with all parameters to be run
    n_cells_arr = np.linspace(n_cells, cells_addition * iLoops, iLoops).astype(np.int)
    n_cells_arr = np.repeat(n_cells_arr, jLoops)
    parameters = itertools.product(
                        *[
                            n_cells_arr,
                            [minutes],
                            [kLoops],
                            [desired_n_samples]
                        ])
    # parameters = [{"n_cells":int(i[0]), "minutes":i[1], "kLoops":i[2], "desired_n_samples":i[3]} for i in parameters]

    parameters = [i for i in parameters]

    # Run the function in a multiprocessed way and produce results
    with mp.Pool(processes=n_cpu) as pool:
        results = pool.map(main_loop_wrapper, parameters)


