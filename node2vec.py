import sys
import multiprocessing
import matplotlib.pyplot as plt
from math import isclose
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter
from IPython.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import stellargraph as sg
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.mapper import FullBatchLinkGenerator, FullBatchNodeGenerator
from stellargraph.layer import Attri2Vec
from stellargraph.layer import GCN, LinkEmbedding
from stellargraph.layer import GraphSAGE
from stellargraph.layer import Node2Vec, link_classification

from tensorflow import keras

def create_biased_random_walker(graph, walk_num, walk_length):
    # parameter settings for "p" and "q":
    p = 1.0
    q = 1.0
    return BiasedRandomWalk(graph, n=walk_num, length=walk_length, p=p, q=q)
# ------------------------------------------------------------------------------

# Get node2vec embedding -------------------------------------------------------
def node2vec_embedding(graph, name, walk_length, batch_size, epochs):

    # Set the embedding dimension and walk number:
    dimension = 128
    walk_number = 20

    # print(f"Training Node2Vec for '{name}':")
    print("Training Node2Vec for " + name + ":")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a Node2Vec training generator, which generates batches of training pairs
    generator = Node2VecLinkGenerator(graph, batch_size)

    # Create the Node2Vec model
    node2vec = Node2Vec(dimension, generator=generator)

    # Build the model and expose input and output sockets of Node2Vec, for node pair inputs
    x_inp, x_out = node2vec.in_out_tensors()

    # Use the link_classification function to generate the output of the Node2Vec model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
    )(x_out)

    # Stack the Node2Vec encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    # Build the model to predict node representations from node ids with the learned Node2Vec model parameters
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = Node2VecNodeGenerator(graph, batch_size).flow(graph_node_list)
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding
# ------------------------------------------------------------------------------

# Train and evaluate the link prediction model ---------------------------------
# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]

# 2. training classifier
def train_link_prediction_model(link_examples, link_labels, get_embedding, binary_operator):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf

def link_prediction_classifier(max_iter=5000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(clf, link_examples_test, link_labels_test, get_embedding, binary_operator):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    roc_score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    ap_score = evaluate_ap(clf, link_features_test, link_labels_test)
    return roc_score, ap_score

# To get ROC score
def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

# To get AP score
def evaluate_ap(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return average_precision_score(link_labels, predicted[:, positive_column])

# We consider 4 different operators: 
    # * *Hadamard*
    # * $L_1$
    # * $L_2$
    # * *average*

def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0

def run_link_prediction(binary_operator, embedding_train, examples_train, labels_train, examples_model_selection, labels_model_selection):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


# Train and evaluate the link model with the specified embedding
def train_and_evaluate(embedding, name, graph_train, examples_train, labels_train, examples_model_selection, labels_model_selection, examples_test, labels_test, walk_length = 5, batch_size = 50, epochs = 10):

    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

    embedding_train = embedding(graph_train, "Train Graph", walk_length, batch_size, epochs)

    # Train the link classification model with the learned embedding
    results = [run_link_prediction(op, embedding_train, examples_train, labels_train, examples_model_selection, labels_model_selection) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])
    print("Best result with " + name + " embeddings from " + best_result['binary_operator'].__name__)
    display(
        pd.DataFrame(
            [(result["binary_operator"].__name__, result["score"]) for result in results],
            columns=("name", "ROC AUC"),
        ).set_index("name")
    )

    # Evaluate the best model using the test set
    test_score = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_train,
        best_result["binary_operator"],
    )

    return test_score


# ===========================================================================================
def run_node2vec(graph_name, graph, rand_state, split_fraction=0.1, walk_length=5, epochs=10, batch_size=50):

    print('Graph:', graph.info())

    # Define an edge splitter on the original graph:
    edge_splitter_test = EdgeSplitter(graph)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
        p=split_fraction, method="global" # original p=0.1
    )

    print('Graph_test:', graph_test.info())

    # Do the same process to compute a training subset from within the test graph
    edge_splitter_train = EdgeSplitter(graph_test)
    graph_train, examples, labels = edge_splitter_train.train_test_split(p=split_fraction, method="global") # original p=0.1
    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25, random_state=rand_state)

    print('Graph_train:', graph_train.info())

    # Below is a summary of the different splits that have been created in this section

    pd.DataFrame(
        [
            (
                "Training Set",
                len(examples_train),
                "Train Graph",
                "Test Graph",
                "Train the Link Classifier",
            ),
            (
                "Model Selection",
                len(examples_model_selection),
                "Train Graph",
                "Test Graph",
                "Select the best Link Classifier model",
            ),
            (
                "Test set",
                len(examples_test),
                "Test Graph",
                "Full Graph",
                "Evaluate the best Link Classifier",
            ),
        ],
        columns=("Split", "Number of Examples", "Hidden from", "Picked from", "Use"),
    ).set_index("Split")


    # Parameter Settings
    walk_length = walk_length # walk_length = 5
    epochs = epochs           # epochs = 10
    batch_size = batch_size   # batch_size = 50

    # Get Node2Vec link prediction result
    print("############ -- Getting node2vec results -- ###############")
    node2vec_result = train_and_evaluate(node2vec_embedding, "Node2Vec", graph_train, examples_train, labels_train, examples_model_selection, labels_model_selection, examples_test, labels_test, walk_length, batch_size, epochs)
    print("############ -- Done node2vec results -- ###############")

    df = pd.DataFrame(
        [
            ("Node2Vec", node2vec_result[0], node2vec_result[1])
        ],
        columns=("name", "ROC score", "AP score"),
    ).set_index("name")

    print(df)
    print("Done calculating node2vec results for " + graph_name + " with random state " + str(rand_state))
    return df
# ===========================================================================================
