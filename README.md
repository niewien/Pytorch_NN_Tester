# Neural Network Performance Tester

This repository contains a Python implementation of a customizable neural network for the classification of MNIST-like data. The model is built using PyTorch and allows flexible experimentation with different hyperparameters such as activation functions, learning rates, hidden layer structures, and more.

## Features

- Modular neural network implementation (`NN` class)
- Support for multiple activation functions
- Hyperparameter tuning (hidden layers, learning rate, loss function, etc.)
- Training and evaluation using PyTorch DataLoader
- Performance logging to a text file
- Saves the best-performing model

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- torchvision

Install the required packages using:

```bash
pip install torch torchvision numpy
```

## Dataset

The code expects a CSV file containing MNIST-like data. The CSV should have:

- Columns 1 to 784: Pixel values (features)
- Column 0: Labels (target values)

Place your dataset file (e.g., `mnist_data_60k.csv`) in the root directory.

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/neural-network-mnist.git
   cd neural-network-mnist
   ```

2. Update the dataset file path in the code:

   ```python
   path = 'mnist_data_60k.csv'
   ```

3. Run the script:

   ```bash
   python main.py
   ```

4. The script will:

   - Train and evaluate the model with multiple configurations.
   - Log performance metrics and hyperparameters to a text file.
   - Save the best-performing model.

## Customization

You can modify the following hyperparameters in the script:

- **Hidden Layers**: Update `hidden_layers` with your desired structures.
- **Learning Rates**: Update `learning_rates`.
- **Activation Functions**: Add or remove functions from the `activations` list.
- **Number of Epochs**: Modify `num_epochs_sets`.
- **Loss Function**: Update the `loss_functions` list.

## Output

- **Logs**: A `model_performance_log_<date>.txt` file is created, storing detailed performance metrics and hyperparameters for each run.
- **Model**: The best-performing model is saved as a `.pth` file.
- **Console Output**: Prints the best model configuration and runtime statistics.

## Example Log

```
Performance: 0.95
Activation function: ReLU()
Learning rate: 0.001
Hidden layer structure: [512]
Number of epochs: 10
Loss function: CrossEntropyLoss()
Batch size: 64
Optimizer: SGD
```

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements or bug fixes.

