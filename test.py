from torch_geometric import datasets
data = datasets.KarateClub()
print(data.num_features, data.num_classes)
print(data._data)