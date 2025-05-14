import pm4py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM
from keras.api.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

def import_xes(file_path):
    log = pm4py.read_xes(file_path)
    event_log = pm4py.convert_to_dataframe(log)

    return event_log

event_log = import_xes("/Users/6706363/Downloads/BPI Challenge 2017.xes")

df = event_log[['case:concept:name', 'concept:name', 'org:resource', 'time:timestamp']]

# Sort by 'time:timestamp' and 'case:concept:name'
df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

df.head(n=10)


def create_activity_resource_sequence(df, prefix_length):
    sequences = []
    grouped = df.groupby('case:concept:name')

    for _, group in grouped:
        activities = group['concept:name'].tolist()
        resources = group['org:resource'].tolist()

        # Only include sequences with length >= prefix_length
        if len(activities) < prefix_length:
            # Remove the sequence (skip appending it to the list)
            continue

        # Truncate to the desired prefix length
        current_activities = activities[:prefix_length]
        current_resources = resources[:prefix_length]  # Include all resources

        # Combine activities and resources into tuples (no changes for the last activity)
        sequence = []
        for i in range(len(current_activities)):
            # For all activities, include both activity and resource
            sequence.append((current_activities[i], current_resources[i]))

        # Add the valid sequence to the list
        sequences.append(sequence)

    return sequences


# Example usage
sequences = create_activity_resource_sequence(df, 5)

# Initialize a set to store unique 'R' values
unique_R = set()

# Loop through the list of sequences and extract the 'R' values
for sequence in sequences:
    for item in sequence:
        # item[1] is the second element (the part with 'R')
        unique_R.add(item[1])

# The length of the set will give the number of unique occurrences of 'R'
print(len(unique_R))

# Prepare the list of activities and resources
activities = []
resources = []

# Loop through sequences to gather activities and resources
for seq in sequences:
    for i, item in enumerate(seq):
        activity, resource = item  # Each item is (activity, resource)
        # Replace NaN resource with 'none'
        if pd.isna(resource):  # Check if the resource is NaN
            resource = 'none'
        activities.append(activity)
        resources.append(resource)

# Fit the OneHotEncoder to the unique activities and resources
activity_encoder = OneHotEncoder()
resource_encoder = OneHotEncoder()

# Fit the encoder on unique activities and resources
activity_encoder.fit([[activity] for activity in set(activities)])
resource_encoder.fit([[resource] for resource in set(resources)])

# Encode activities and resources
encoded_sequences = []
y_encoded = []  # List to store the one-hot encoded target resource for the last activity

for seq in sequences:
    activity_onehots = []

    # For each activity-resource pair, apply one-hot encoding
    for i, item in enumerate(seq):
        activity, resource = item
        # Replace NaN resource with 'none' during encoding
        if pd.isna(resource):  # Check if the resource is NaN
            resource = 'none'
        activity_onehot = activity_encoder.transform([[activity]]).toarray()

        # If it's the last item, we only encode the activity and store the resource for y
        if i == len(seq) - 1:
            # Add only the activity one-hot encoding
            activity_onehots.append(activity_onehot)
            # One-hot encode the resource and store it for prediction (y)
            resource_onehot = resource_encoder.transform([[resource]]).toarray()
            y_encoded.append(resource_onehot)  # Store the one-hot encoded resource
        else:
            # For all other activities, include both activity and resource one-hot encoding
            resource_onehot = resource_encoder.transform([[resource]]).toarray()
            encoded_sequence = np.hstack([activity_onehot, resource_onehot])
            activity_onehots.append(encoded_sequence)

    # If there is more than one activity in the sequence, add the zero vector for the last resource
    if len(seq) > 1:
        last_activity_onehot = activity_onehots[-1]
        last_resource_onehot = np.zeros(resource_onehot.shape)  # Zero vector for the last resource
        activity_onehots[-1] = np.hstack([last_activity_onehot, last_resource_onehot])

    # Concatenate the encoded activities and resources for the full sequence
    encoded_sequences.append(np.vstack(activity_onehots))

X = np.array(encoded_sequences)
y = np.array(y_encoded)

import matplotlib.pyplot as plt

# Initialize KFold with 5 splits
kf = KFold(n_splits=5)


# Initialize the model
def create_model():
    model = Sequential()
    # First LSTM layer with return_sequences=True
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    # Second LSTM layer
    model.add(LSTM(50))
    # Output Dense layer
    model.add(Dense(118, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Store metrics from each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Initialize EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Initialize list to store loss curves
training_losses = []
validation_losses = []

# Loop through the KFold splits
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Squeeze the target arrays to remove the extra dimension
    y_train = y_train.squeeze(axis=1)
    y_test = y_test.squeeze(axis=1)

    # Create the model for each fold
    model = create_model()

    # Train the model with early stopping
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Capture the training and validation losses
    training_losses.append(history.history['loss'])
    validation_losses.append(history.history['val_loss'])

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    y_test_classes = np.argmax(y_test, axis=1)  # Ensure test labels are in class label format

    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_test_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)

    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Output average metrics
print(f'Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
print(f'Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}')
print(f'Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
print(f'Average F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')

# Plot the loss curve after all folds
plt.figure(figsize=(10, 6))

# Plot the loss for each fold
for i in range(len(training_losses)):
    plt.plot(training_losses[i], label=f'Train Fold {i + 1}')
    plt.plot(validation_losses[i], label=f'Validation Fold {i + 1}')

plt.title('Loss Curve for Each Fold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()






