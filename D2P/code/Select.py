import pickle
import os
import json
import numpy as np
from core.data.Coreset import CoresetSelection

output = []
for epoch in range(1, 6):
    td_path = os.path.join("/D2P/D2P_Round1_Model", f"td_{epoch}.pkl")
    with open(td_path, 'rb') as f:
        pickle_data = pickle.load(f)
    td_log = pickle_data["training_dynamics"]
    output.append([data_point["output"].item() for data_point in td_log])

output = np.array(output)
variance = np.std(np.array(output), axis=0)
print(variance.shape)


feature_path = "/D2P/train_features.npy"
data_embeds = np.load(feature_path)

coreset_index = CoresetSelection.stratified_sampling(variance, coreset_num=1000, data_embeds=data_embeds, n_neighbor=100, stratas=100)
np.save("/D2P/Coreset_Index.npy", coreset_index)


with open('/home/hywang/projects/d2pruning/train.json', 'r') as f:
    train_data = json.load(f)

selected_data = [train_data[i] for i in coreset_index]

with open('/D2P/coreset.json', 'w') as f:
    json.dump(selected_data, f, indent=4)
