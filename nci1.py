## Example script for basic useage of WL+KSVD method
# Following the example by : https://karateclub.readthedocs.io/en/latest/notes/introduction.html#graph-embedding

# Import packages
from karateclub.dataset import GraphSetReader
from karateclub import FeatherGraph, Graph2Vec, GL2Vec,FGSD, SF, NetLSD, GL2Vec, GeoScattering, IGE, LDP, WaveletCharacteristic
from WL_KSVD import WL_KSVD

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
import os
import networkx as nx
from rdkit import Chem

# Load data
print('Loading NCI1 dataset')

DATASET_DIR = "datasets/NCI_full"  # change this
graphs = []
y = []

for filename in os.listdir(DATASET_DIR):
    if filename.endswith(".sdf"):
        filepath = os.path.join(DATASET_DIR, filename)

        supplier = Chem.SDMolSupplier(filepath, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue

            G = nx.Graph()

            # Add atoms as nodes
            for atom in mol.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    label=atom.GetSymbol()   # WL uses node labels
                )

            # Add bonds as edges
            for bond in mol.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx()
                )

            # Get graph label
            # In NCI1, class label is stored as a molecule property
            label = int(float(mol.GetProp("value")))
            graphs.append(G)


            y.append(label)

print(f"Loaded {len(graphs)} graphs")

# Fit the WL+KSVD model
print('Fitting the embedding model')
# model = FeatherGraph() # Test with other whole-graph embedding methods in karateclub package
model = WL_KSVD()

model.fit(graphs)
X = model.get_embedding()

# Train Test divide
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying ML model
print('Applying ML model')

downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
print('Without overfit protection')
print('AUC: {:.4f}'.format(auc))

#############################################################################################################
# Over-fit protection
print('Over-fit protection')
# To avoid over-fitting we can fit the WL+KSVD model on the training set and infer the embedding of the test set.
# First we divide the data into train and test sets.

G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)

# divide the train set further into vocab training and ML training sets

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75, random_state=42)

# Fit the WL+KSVD model on the vocab training set
print('Fitting the embedding model')
model = WL_KSVD()
# model = Graph2Vec()

model.fit(G_vocab_train)
X_vocab_train = model.get_embedding()

# Infer the embedding of the ML training set
X_ML_train = model.infer(G_ML_train)
X_ML_test = model.infer(G_test)

# Scaling the embedding such that sparsity is preserved
scaler = MaxAbsScaler()
X_ML_train_scaled = scaler.fit_transform(X_ML_train)

X_ML_test_scaled = scaler.transform(X_ML_test)

# Applying ML model
print('Applying ML model')
downstream_model = LogisticRegression(random_state=0).fit(X_ML_train_scaled, y_ML_train)
y_ML_hat = downstream_model.predict_proba(X_ML_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_ML_hat)
print('With over-fit protection')
print('AUC: {:.4f}'.format(auc))
print('Done')
