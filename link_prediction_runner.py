# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import scipy.sparse 
from scipy.sparse import csr_matrix
import pickle
import stellargraph as sg
import os
from stellargraph import StellarGraph, datasets

from math import isclose
from sklearn.decomposition import PCA

from node2vec import run_node2vec


# ================================================================================================================================================================
# Make the graph from adj
def get_sg_graph(adj):
    print('adj shape:', adj.shape)
    nxGraph = nx.from_scipy_sparse_array(adj)                           # make nx graph from scipy matrix

    # make StellarGraph from nx graph
    sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="gene", edge_type_default="link")
    print(sgGraph.info())

    return sgGraph
# ================================================================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--folder_name')
parser.add_argument('--dataset')

# python comparison_runner.py --folder_name=kegg --dataset=hsa04151
args = parser.parse_args()
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Set up
folder_name = args.folder_name # 'kegg'
data_name = args.dataset       # 'hsa04151' or 'hsa04151_unique'
split_num = [42]               # split_num = [42, 56, 61, 69, 73]
print('Running node2vec for ' + data_name)

# ================================================================================================================================================================
# read adj from pickle and prepare sg graph
adjPickleFile = '../graph-data/{}/Processed/{}_adj.pickle'.format(folder_name, data_name)
with open(adjPickleFile, 'rb') as handle: adj = pickle.load(handle) 

sgGraph = get_sg_graph(adj)        # make the graph

# run Node2Vec model for sg graph
for rand_state in split_num:

    outputDf = run_node2vec(data_name, sgGraph, rand_state, split_fraction=0.1)

    outputFileName = "results/{}_{}.txt".format(data_name, rand_state)
    f1 = open(outputFileName, "w")
    f1.write("For data_name: {}, split: {}\n".format(data_name, rand_state))
    f1.write(outputDf.to_string())
    f1.close()
    print("Done calculating node2vec results for " + data_name + " with random state " + str(rand_state))
# ================================================================================================================================================================

# dataset --> split_fraction (p)
# hsa04740 --> 0.001
# hsa05168 --> 0.01
# hsa04151 --> 0.1
# hsa04010 --> 0.1
# hsa04014 --> 0.1