import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np


# Class declarations
class NN(nn.Module):
    # Initialize the neural network with parameters for input size, output size, hidden layers, and activation function
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation):
        super(NN, self).__init__()

        self.Sequential = nn.Sequential()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.network = []  # List to hold layers
        self.update_network(hidden_layer_sizes, activation)  # Build network based on hidden layer sizes and activation function

    # Forward pass through the network (flatten input and then pass through network layers)
    def forward(self, x):
        """
        Flatten the input image data and pass it through the network layers.
        The input is a 2D image of size 28x28, which is flattened into a 1D vector of size 784.
        The vector is passed through the neural network layers, and logits are returned.

        Arguments:
        x -- The input data, a tensor representing an image.

        Returns:
        logits -- The raw outputs of the model (before activation) used for classification.
        """
        x = self.flatten(x)  # Flatten input (28x28 pixels to 784 features for MNIST)
        logits = self.network(x)  # Pass through the layers
        return logits

    # Train the model using a specified loss function, optimizer, number of epochs, and training data
    def train_model(self, loss_function, optimizer, num_epochs, train_data, lr):
        """
        This function trains the model using gradient descent with backpropagation.
        It iterates over the training data for the specified number of epochs, computes the loss, and updates the model parameters.

        Arguments:
        loss_function -- The loss function used to compute the error (e.g., CrossEntropyLoss).
        optimizer -- The optimization algorithm used for weight updates (e.g., SGD).
        num_epochs -- The number of epochs for training.
        train_data -- The training data as a DataLoader object.
        lr -- The learning rate used by the optimizer.

        Returns:
        None
        """
        self.train()  # Set the model to training mode
        for epoch in range(num_epochs):  # Loop over epochs
            for idx, (data, labels) in enumerate(train_data):  # Loop over batches in the training data
                data, labels = data.to(device), labels.to(device)  # Move data and labels to the correct device
                labels = labels.type(torch.LongTensor)  # Convert labels to long tensor type

                # Feed data through the network
                outputs = self(data)
                loss = loss_function(outputs, labels)  # Compute the loss based on output and true labels

                # Back propagation
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights with gradient descent

                # Print the loss every 100 steps
                if idx % 100 == 0:
                    loss, current = loss.item(), idx * len(data)
                    print(f"Loss: {loss} for run {current} in epoch {epoch + 1} with activation function {self.activation}"
                          f" with learn rate {lr} and loss function {loss_function}.")

    # Evaluate the model performance on the test dataset
    def evaluate_performance(self, test_data, lr, epochs, lf):
        """
        This function evaluates the model's performance by calculating accuracy on the test dataset.

        Arguments:
        test_data -- The test dataset (DataLoader).
        lr -- Learning rate used during training.
        epochs -- Number of epochs used during training.
        lf -- Loss function used during training.

        Returns:
        performance -- The accuracy of the model on the test dataset.
        """
        self.eval()  # Set the model to evaluation mode
        right_classifications = 0  # Count of correct classifications
        with torch.no_grad():  # No need to compute gradients during evaluation
            for idx, (data, label) in enumerate(test_data):  # Loop over test data batches
                data, label = data.to(device), label.to(device)  # Move data and labels to the correct device
                output = model(data)  # Get model predictions
                result = output.argmax()  # Get the predicted class with highest probability

                # If prediction is correct, increment count
                if result == label:
                    right_classifications += 1
        performance = right_classifications / len(test_loader)  # Compute accuracy
        print(f"Performance: {performance} \n"
              f"Activation function: {self.activation} \n" 
              f"Learning rate: {lr} \n"
              f"Hidden layers: {self.hidden_layer_sizes} \n"
              f"Number of epochs: {epochs} \n"
              f"Loss function: {lf}")
        print("\n")
        return performance

    # Update the network based on new hidden layer sizes or activation function
    def update_network(self, new_hidden_layer_sizes, new_activation):
        """
        This function updates the model's architecture by changing the hidden layer sizes or activation function.

        Arguments:
        new_hidden_layer_sizes -- The new sizes for the hidden layers.
        new_activation -- The new activation function.

        Returns:
        None
        """
        # Check if the hidden layer sizes need to be updated
        if new_hidden_layer_sizes != self.hidden_layer_sizes:
            self.hidden_layer_sizes = new_hidden_layer_sizes  # Update hidden layer sizes
            self.__build_network()  # Rebuild the network
            print(f"Updated network to {self.activation} activation function .")
        # Check if the activation function needs to be updated
        if new_activation != self.activation:
            self.activation = new_activation  # Update activation function
            self.__build_network()  # Rebuild the network
            print(f"Updated network to {new_hidden_layer_sizes} hidden layers.")

    # Build the neural network model layers
    def __build_network(self):
        """
        This function builds the network layers (input layer, hidden layers, output layer) based on the provided
        configuration of hidden layer sizes and activation function.

        Arguments:
        None

        Returns:
        None
        """
        layers = []  # List to hold layers of the network

        # Input layer to first hidden layer
        layers.append(nn.Linear(self.input_size, self.hidden_layer_sizes[0]))  # input -> first hidden layer
        layers.append(self.activation)  # Add activation after input

        # Hidden layers
        for i in range(1, len(self.hidden_layer_sizes)):
            layers.append(nn.Linear(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i]))  # Hidden layer
            layers.append(self.activation)  # Add activation after each hidden layer

        # Output layer
        layers.append(nn.Linear(self.hidden_layer_sizes[-1], self.output_size))  # Last hidden to output

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)  # Sequential model containing all layers


class Data(Dataset):
    # Initialize the dataset by reading a CSV file, loading input and target data
    def __init__(self):
        """
        This function loads the dataset from a CSV file, extracts input features and target labels,
        and converts them into PyTorch tensors.

        Arguments:
        None

        Returns:
        None
        """
        xy = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=1)  # Load CSV (MNIST data)
        self.x = torch.from_numpy(xy[:, 1:])  # Input features (all columns except the first)
        self.y = torch.from_numpy(xy[:, 0])  # Target labels (first column)
        self.n_samples = xy.shape[0]  # Total number of samples

    # Get a sample from the dataset by index
    def __getitem__(self, index):
        """
        This function retrieves a data sample (input and label) from the dataset at a specified index.

        Arguments:
        index -- The index of the sample to retrieve.

        Returns:
        A tuple containing the input features and the target label.
        """
        return self.x[index], self.y[index]

    # Return the total number of samples in the dataset
    def __len__(self):
        """
        This function returns the total number of samples in the dataset.

        Arguments:
        None

        Returns:
        The total number of samples in the dataset.
        """
        return self.n_samples


def log_performance_to_file(file_path, performance, activation_fn, learning_rate, hidden_layer, num_epochs, loss_function, batch_size, optimizer_type):
    """
    This function logs the performance and the hyperparameters used to a text file.

    Arguments:
    file_path -- The file path to save the log.
    performance -- The performance (accuracy) achieved by the model.
    activation_fn -- The activation function used in the model.
    learning_rate -- The learning rate used during training.
    hidden_layer -- The hidden layer structure used in the model.
    num_epochs -- The number of epochs used during training.
    loss_function -- The loss function used during training.
    batch_size -- The batch size used during training.
    optimizer_type -- The type of optimizer used (e.g., 'SGD', 'Adam').

    Returns:
    None
    """
    with open(file_path, 'a') as f:
        f.write(f"Performance: {performance}\n"
                f"Activation function: {activation_fn}\n"
                f"Learning rate: {learning_rate}\n"
                f"Hidden layer structure: {hidden_layer}\n"
                f"Number of epochs: {num_epochs}\n"
                f"Loss function: {loss_function}\n"
                f"Batch size: {batch_size}\n"
                f"Optimizer: {optimizer_type}\n\n")


# App
if __name__ == "__main__":

    # Constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically use GPU if available
    print(f"Using device: {device}")  # Print selected device (CPU or GPU)

    path = r'mnist_data_60k.csv'
    save_path = f'model_performance_log_{datetime.date.today()}.txt'
    input_size = 784
    output_size = 10
    train_ratio = 0.8
    batch_size = 64

    # Evaluation parameters
    num_epochs_sets = [5, 10]
    hidden_layers = [[512], [300]]
    learning_rates = [0.0001, 0.001]
    activations = [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01), nn.PReLU(), nn.ELU(alpha=0.1)]
    loss_functions = [nn.CrossEntropyLoss()]

    # Set test and training set
    dataset = Data()
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader instances for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    # Main body
    # Runtime start
    start = time.time()

    # Initialize Neural Network
    model = NN(input_size, output_size, hidden_layers[0], nn.Sigmoid()).to(device)  # Move model to device

    # Evaluate best performance
    best_performance = 0
    best_af = -1
    best_lr = -1
    best_hl = -1
    best_ne = -1
    best_lf = -1
    for i, activation_fn in enumerate(activations):
        for j, learning_rate in enumerate(learning_rates):
            for k, hidden_layer in enumerate(hidden_layers):
                for x, num_epochs in enumerate(num_epochs_sets):
                    for y, loss_function in enumerate(loss_functions):
                        # Switch Activation
                        model.update_network(hidden_layer, activation_fn)

                        # Set Optimizer, switch learning rate
                        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                        # Training
                        model.train_model(loss_function, optimizer, num_epochs, train_loader, learning_rate)

                        # Evaluate Performance
                        performance = model.evaluate_performance(test_loader, learning_rate, num_epochs, loss_function)
                        if performance > best_performance:
                            best_performance = performance
                            best_af = i
                            best_lr = j
                            best_hl = k
                            best_ne = x
                            best_lf = y

                        # Log performance
                        log_performance_to_file(save_path, performance, activation_fn, learning_rate, hidden_layer, num_epochs,
                                                loss_function, batch_size, optimizer)

    print(f"Best performance is {best_performance} with \n"
          f"Activation function: {activations[best_af]} \n"
          f"Learn rate: {learning_rates[best_lr]} \n"
          f"Hidden layer structure: {hidden_layers[best_hl]} \n"
          f"Number of epochs: {num_epochs_sets[best_ne]} \n"
          f"Loss function: {loss_functions[best_lf]} \n")

    # Save the model
    model_save_path = (f"model_with_{model.activation}_"
                       f"{learning_rates[best_lr]}_lr_"
                       f"{hidden_layers[best_hl]}_hidden_layer_structure_"
                       f"{num_epochs_sets[best_ne]}_number_of_epochs_"
                       f"{loss_functions[best_lf]}_loss_function_"
                       f"{batch_size}_bach_size.pth")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Calculate and print Runtime
    end = time.time()
    print(f"Runtime:  {end-start}, s")