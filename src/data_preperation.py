import numpy as np
import pandas as pd
import argparse
import networkx as nx
from networkx.readwrite import json_graph
import json
import random
import psycopg2
from joblib import Parallel, delayed
from collections import OrderedDict

"""
Input: dataframe with columns 
            "id" - name of te edge
            "source" - node where the edge starts 
            "target" - node where the edge ends
            
Output: Line Graph as dataframe with columns
            "source" - node where the edge starts 
            "target" - node where the edge ends
"""

def construct_graph(df):
    df[["id", "source", "target"]].copy()

    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row["source"], row["target"], id=row["id"])

    print("Graph created with " + str(G.number_of_nodes()) + " nodes and " + str(G.number_of_edges()) + " edges.")
    return G


def create_id_map_json(df):
    edge_list = df['id'].tolist()
    id_map = {edge_list[i]: i for i in range(0, len(edge_list))}
    return id_map


# Saves a Dictionary as JSON File
def save_json(file, path):
    with open(path, 'w') as fp:
        json.dump(file, fp)
    print("JSON saved: " + path)


# This Function creates a Feature Matrix with features: MAX SPEED, LENGTH, TYPE
# Those features are static for the Graph and therefor we need only one instance for all timesteps
def createFeatureMatrix(df):
    # We need to use all feature wich come in
    # So drop those wich we do not need/ can not use
    feature_df = df.drop(['id', 'time', 'source', 'target', 'speed'], axis=1)
    # feature_df = df[["max_speed", "length", "type"]].copy()

    # One-Hot-Encode Categorial Variables
    feature_df_single_encoded = pd.get_dummies(feature_df, columns=["type"])
    # Transform to Matrix
    feature_mx = feature_df_single_encoded.values

    return feature_mx


# Get only one timestamp with the most streetIDs
def get_single_timestamp(df):
    # As not every timestamp is complete take the timestamp with the most streetIDs
    df_grouped = df.groupby('time')
    idx_max_graph = df_grouped['id'].count().idxmax()
    single_df = df_grouped.get_group(idx_max_graph)
    
    # Remove rows where source = target
    indexes_duplicates = single_df.loc[single_df["source"] == single_df["target"]].index
    single_df = single_df.drop(indexes_duplicates)
    
    return single_df

def returnNotMatches(a, b):
    return [x for x in a if x not in b]

def createClassMap(df, streetgraph_df):
    # labels_df_grouped = labels_df.groupby('time')
    #
    # mx_list = []
    # for name, group in labels_df_grouped:
    #     mx_list.append(dict(zip(group.id, group.speed)))

    # Problem is, that the extracted streetgraph  can be incomplete
    # First fix to use the complete streetgraph of all cities resulted in worse performance
    # Therefor here we remove all labels which are not represented in the graph
    # Check if all ids exist in streetgraph, if not remove
    df_id = df[['id']].drop_duplicates()
    missing_ids = returnNotMatches(list(df_id.values), list(streetgraph_df['id']))
    print("IDs not in streetgraph: " + str(missing_ids))
    for id in missing_ids:
        df = df[df.id != id[0]]

    # A Label is identified by the Node ID and the Timestep
    label_df = df[["id","time","speed"]].copy()
    #label_df.columns = ["id","timestep","speed"]

    # Important: The timesteps should be int from 0 to n and map exactly to the row values of the timefeats matrix.
    distinct_timesteps =df.loc[:,['time']].copy()
    distinct_timesteps = distinct_timesteps.drop_duplicates()
    distinct_timesteps["timestep"] = np.arange(len(distinct_timesteps))
    # Merge timesteps to DF
    label_df = pd.merge(label_df, distinct_timesteps, how = 'left', left_on="time", right_on="time")
    label_df = label_df.drop('time', 1)

    # Rearrange & Rename
    label_df = label_df[["id","timestep","speed"]]
    label_df.columns = ["id", "timestep", "label"]
    return label_df

def create_timefeat_mx(df):
    df = df.loc[:,['time']]
    df = df.drop_duplicates()
    # From Datetime we need dayofweek and timeofday
    df['weekday'] = df['time'].dt.dayofweek
    df['timeofday'] = (df['time'].dt.hour + df['time'].dt.minute/60) / 24
    df = df.drop('time', 1)
    # One-Hot-Encode Categorial Variables
    df = pd.get_dummies(df, columns=["weekday"])
    return df.values


def createInputs(df):

    single_df = get_single_timestamp(df)

    # Create Line Graph from Graph
    G = construct_graph(single_df)
    #df.to_csv("graph.csv")

    # Create Inputfile: id-map.json
    # It is important that the order will be the same as for the feature Matrix
    id_map_dict = create_id_map_json(single_df)

    # Create Inputfile: feats.npy
    feature_mx = createFeatureMatrix(single_df)

    # Create Inputfile: class_map.json
    # Here we need all timestamps as Labels vary for every timestamp
    label_df = createClassMap(df, single_df)


    timefeat_mx = create_timefeat_mx(df)

    return G, feature_mx, id_map_dict, label_df, timefeat_mx



def normalize_input(feature_mx, label_df, train_test = 'train', scaler=None, scaler_y = None):
        if train_test == 'train':
            from sklearn.preprocessing import StandardScaler
            #  Train Scaler on train set and use it for testing as well

            scaler = StandardScaler()
            scaler.fit(feature_mx)
            feature_mx = scaler.transform(feature_mx)

            # Stefan: Standerdize Labels as well
            # Training: get labels as 1-Dim Array
            np_labels = label_df["label"].values
            # Fit Scaler
            scaler_y = StandardScaler()
            scaler_y.fit(np_labels.reshape(-1, 1))
            np_labels_norm = scaler_y.transform(np_labels.reshape(-1, 1))
            label_df["label"] = np_labels_norm
        else:
            feature_mx = scaler.transform(feature_mx)

            np_labels = label_df["label"].values
            np_labels_norm = scaler_y.transform(np_labels.reshape(-1, 1))
            label_df["label"] = np_labels_norm

        return feature_mx, label_df, scaler, scaler_y
