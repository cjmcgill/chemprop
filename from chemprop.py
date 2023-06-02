from chemprop.data import MoleculeDataset, MoleculeDataLoader
from chemprop.models import build_model
from chemprop.train import train, evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Assume you have loaded the QM9 dataset into a file called 'qm9.csv'

# Load the dataset using Chemprop
dataset = MoleculeDataset('qm9.csv')

# Split the dataset into an initial labeled set and an unlabeled set
train_data, unlabeled_data = train_test_split(dataset, test_size=0.8, random_state=42)

# Define the active learning parameters
initial_labeled_samples = 100  # Number of initially labeled samples
query_samples = 50  # Number of samples to query at each iteration
num_iterations = 5  # Number of active learning iterations

# Initialize the labeled set with random samples
labeled_data, unlabeled_data = train_test_split(train_data, train_size=initial_labeled_samples, random_state=42)

# Initialize the model
model = build_model()

# Active learning loop
for iteration in range(num_iterations):
    # Train the model on the labeled data
    train_data_loader = MoleculeDataLoader(labeled_data)
    train(model, train_data_loader)

    # Make predictions on the unlabeled data
    unlabeled_data_loader = MoleculeDataLoader(unlabeled_data)
    predictions = evaluate(model, unlabeled_data_loader)

    # Calculate the uncertainty (e.g., MSE) for each prediction
    uncertainties = np.abs(predictions - unlabeled_data.targets())

    # Select the most informative samples based on uncertainty
    query_indices = uncertainties.argsort()[:query_samples]

    # Query the true labels for the selected samples (e.g., through experiments or expert knowledge)
    queried_data = unlabeled_data.select(query_indices)

    # Add the queried samples to the labeled set
    labeled_data.extend(queried_data)

    # Remove the queried samples from the unlabeled set
    unlabeled_data = unlabeled_data.remove(query_indices)

# Final model training on all labeled samples
train_data_loader = MoleculeDataLoader(train_data)
train(model, train_data_loader)

# Evaluate the final model's performance
test_data_loader = MoleculeDataLoader(dataset)
predictions = evaluate(model, test_data_loader)
targets = dataset.targets()
rmse = np.sqrt(mean_squared_error(targets, predictions))
print("Final RMSE:", rmse)
