import joblib
import pickle
import numpy as np
import csv

# The calling path must match
scaler_path = '.pkl'
model_path = '.pkl'

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded.")
model = joblib.load(model_path)
print("XGB model loaded.")

# Dataset to be predicted
data_path = 'input/Edong.csv'

data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
features = data[:, 1:] # Corresponding features

new_data_scaled = scaler.transform(features)

# Output
output_csv_path = '.csv' # Path
predicted_label = model.predict(new_data_scaled)
header = ["SIO2", "TIO2", "AL2O3", "FEOT", "MNO", "MGO", "CAO", "CR2O3", "PRE_LABEL"]
rows = []
for i in range(features.shape[0]):
    row = list(features[i])
    row.append(predicted_label[i])
    rows.append(row)

with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)
