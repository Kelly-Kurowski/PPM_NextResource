import pm4py
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def import_xes(file_path):
    log = pm4py.read_xes(file_path)
    event_log = pm4py.convert_to_dataframe(log)

    return event_log

event_log = import_xes("/Users/6706363/Downloads/BPI_Challenge_2013_incidents.xes")

# roles = pm4py.discover_organizational_roles(
#     event_log,
#     resource_key='org:resource',
#     activity_key='concept:name',
#     timestamp_key='time:timestamp',
#     case_id_key='case:concept:name'
# )
#
# print(roles)

activity_matrix = pd.crosstab(event_log['org:resource'], event_log['concept:name'])
print(activity_matrix)

print(event_log)

scaler = MinMaxScaler()
activity_matrix_normalized = scaler.fit_transform(activity_matrix)

n_clusters = 4  # Number of roles
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    activity_matrix_normalized.T,  # Transpose so each activity is a feature
    c=n_clusters,                 # Number of clusters (roles)
    m=2.5,                          # Fuzziness coefficient
    error=0.005,                  # Convergence error tolerance
    maxiter=1000,                 # Maximum number of iterations
    init=None,                    # Initial cluster centers (optional)
    seed=42                       # Random seed for reproducibility
)

print(u)


plt.imshow(u, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Membership Degree")
plt.title("Fuzzy C-Means Membership Matrix")
plt.xlabel("Resources")
plt.ylabel("Roles")
plt.show()
