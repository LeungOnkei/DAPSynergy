import csv
from itertools import islice
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import networkx as nx
from utils_test import *


def get_cell_feature(cellId, cell_features):

    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1:]


def get_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(1024)

    # Morgan (ECFP4)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

    return np.array(morgan_fp, dtype=np.float32)


def get_physicochemical_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(8)

    properties = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.HeavyAtomCount(mol)
    ]

    return np.array(properties, dtype=np.float32)


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, [], [], np.zeros(1024), np.zeros(8)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    morgan_fp = get_morgan_fingerprint(smile)
    physchem_props = get_physicochemical_properties(smile)

    return c_size, features, edge_index, morgan_fp, physchem_props


def creat_data(datafile, cellfile):
    cell_features = []
    with open(cellfile) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    #print('cell_features_shape:', cell_features.shape)


    compound_iso_smiles = []
    df = pd.read_csv('data/smiles.csv')
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)

    smile_graph = {}
    smile_fingerprints = {}
    smile_properties = {}

    for i, smile in enumerate(compound_iso_smiles):
        if i % 100 == 0:
            print(f'processed {i}/{len(compound_iso_smiles)} ')

        g = smile_to_graph(smile)
        smile_graph[smile] = (g[0], g[1], g[2])
        smile_fingerprints[smile] = g[3]
        smile_properties[smile] = g[4]


    processed_data_file_train = 'data/processed/' + datafile + '_train.pt'

    if not os.path.isfile(processed_data_file_train):
        df = pd.read_csv('data/' + datafile + '.csv')
        drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)


        TestbedDataset(
            root='data',
            dataset= datafile + '_drug1',
            xd=drug1,
            xt=cell,
            xt_feature=cell_features,
            y=label,
            smile_graph=smile_graph,
            smile_fingerprints=smile_fingerprints,
            smile_properties=smile_properties
        )

        print('drug1 done！')


        TestbedDataset(
            root='data',
            dataset= datafile + '_drug2',
            xd=drug2,
            xt=cell,
            xt_feature=cell_features,
            y=label,
            smile_graph=smile_graph,
            smile_fingerprints=smile_fingerprints,
            smile_properties=smile_properties
        )

        print('drug2 done！')


if __name__ == "__main__":
    cellfile = 'data/cell_features_954.csv'
    da = ['new_labels_0_10']

    for datafile in da:
        print(f'processing: {datafile}')
        creat_data(datafile, cellfile)