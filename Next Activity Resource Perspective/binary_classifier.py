from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Calculate the resource activity matrix
def create_diversity_matrix(log):

    activity_counts = log.pivot_table(index='org:resource',
                                  columns='concept:name',
                                  aggfunc='size',
                                  fill_value=0)

    # Resetting the index for a cleaner look
    activity_counts.reset_index(inplace=True)

    return activity_counts

def train_random_forest_with_randomized_search(df_encoded, activities):
    label_encoders = {}  # Store encoders for decoding later

    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le  # Save the encoder for decoding predictions

    # Define the hyperparameter grid for Randomized Search
    param_distributions = {
        'n_estimators': np.arange(1,101),  # Number of trees in the forest
        'max_depth': np.arange(1,51),  # Maximum depth of the trees
        'min_samples_split': np.arange(1,51),  # Minimum samples required to split an internal node
        'min_samples_leaf': np.arange(1,51),  # Minimum samples required to be at a leaf node
        'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
        'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider for the best split
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }

    rf = RandomForestClassifier()

    # Initialize a variable to store the sum of accuracies for all activities
    sum_acc = 0
    results = {}
    best_params_list = []  # List to store best parameters for each activity

    # Perform Randomized Search for each activity and find the best params
    for activity in activities:
        print(f"\nTuning Random Forest for activity: {activity}")

        # Prepare the data: X (features) and y (target)
        X = df_encoded.drop(activity, axis=1)  # Drop the current activity column
        y = df_encoded[activity]  # Use the current activity as the target

        # Train-test split (this will be specific to each activity)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Perform Randomized Search with cross-validation
        randomized_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=20,  # Number of parameter settings sampled
            cv=3,  # 3-fold cross-validation
            n_jobs=-1,  # Use all available CPUs
            verbose=1
        )

        # Fit the Randomized Search on training data
        randomized_search.fit(X_train, y_train)

        # Get the best parameters for this activity
        best_params = randomized_search.best_params_
        best_params_list.append(best_params)  # Store the best parameters
        print(f"Best parameters for {activity}: {best_params}")

        # Train the final model with the best hyperparameters for this activity
        best_rf = randomized_search.best_estimator_

        # Predict on the test set
        y_pred = best_rf.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        sum_acc += accuracy

        # Store the accuracy result for the current activity
        results[activity] = accuracy
        print(f"{activity} has accuracy: {accuracy:.2f}")

    # Calculate the average accuracy across all activities
    final_average_accuracy = sum_acc / len(activities)
    results['Average Accuracy'] = final_average_accuracy

    print(f"\nFinal Average Accuracy across all activities: {final_average_accuracy:.3f}")

    # Count occurrences of each parameter value across all activities
    param_counts = {key: Counter() for key in param_distributions.keys()}  # Initialize a Counter for each parameter

    for params in best_params_list:
        for key, value in params.items():
            param_counts[key][value] += 1  # Count occurrences of each value for each parameter

    # Find the most common value for each parameter
    most_common_params = {key: count.most_common(1)[0] for key, count in param_counts.items()}

    print("\nMost Common Parameter Values Across All Activities:")
    for param, (value, count) in most_common_params.items():
        print(f"{param}: {value} (Count: {count})")

    trained_models = {}  # Dictionary to store the trained classifiers for each activity

    for activity in activities:
        # Train a new model with the best hyperparameters
        best_rf = RandomForestClassifier(**best_params_list[activities.index(activity)])
        X = df_encoded.drop(activity, axis=1)
        y = df_encoded[activity]
        best_rf.fit(X, y)  # Train on full dataset
        trained_models[activity] = best_rf  # Store the trained model

    return trained_models, X_train  # Return all trained classifiers