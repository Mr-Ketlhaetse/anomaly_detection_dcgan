try:
    from sklearn.base import BaseEstimator
    import torch
    import random
    import torch.nn as nn
    import pandas as pd
    from config import Parameters as prm
except ImportError as e:
    print(f"Import error: {e}")
    raise


class CNNTransferLearning(BaseEstimator):
    def __init__(self, dcgan_discriminator, num_classes, learning_rate=0.001):
        super(CNNTransferLearning, self).__init__()

        # Load weights from the trained discriminator
        self.dcgan_discriminator = dcgan_discriminator
        # Freeze the discriminator weights
        for param in self.dcgan_discriminator.parameters():
            param.requires_grad = False

        # Additional layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(180, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

        self.model = nn.Module()  # Create an nn.Module instance
        self.model = nn.Sequential(
            self.dcgan_discriminator,
            self.classifier
        )
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def forward(self, x):
        x = self.dcgan_discriminator(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def fit(self, X, y, learning_rate=None):
        # Ensure that X is a list of torch tensors
        X = [x.to(dtype=torch.float32) for x in X]
        # Check if y is a list of floats or a Series
        # Check if y is a list of floats or a Series
        if isinstance(y, list) and y and isinstance(y[0], float):
            y = torch.tensor(y).long()
        elif isinstance(y, pd.Series):
            y = torch.tensor(y.values).long()
        elif isinstance(y, list):
            y = [y_i.to(dtype=torch.long) for y_i in y]

        # Set the model to training mode
        self.model.train()

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate if learning_rate is None else learning_rate)

        # Train the model
        for epoch in range(10):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.forward(X)
            # X_tensor = torch.stack(X) if isinstance(X, list) else X  # Convert X to a tensor if it's a list
            y_tensor = torch.stack(y) if isinstance(y, list) else y  # Convert y to a tensor if it's a list

            loss = criterion(outputs, y_tensor)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        history = {
            'loss': loss.item(),
            'accuracy': (torch.argmax(outputs, dim=1) == y_tensor).sum().item() / len(y_tensor)
            }

        return self, history

    def predict(self, X):
        # Convert X to torch tensor
        X = torch.stack([x.to(dtype=torch.float32) for x in X])

        # Set the model to evaluation mode
        self.model.eval()

        # Forward pass
        outputs = self.forward(X)

        # Get the predicted class indices
        _, predicted_indices = torch.max(outputs, 1)

        # Convert predicted indices to numpy array
        predictions = predicted_indices.detach().numpy()

        return predictions

    def score(self, X):
        # Convert X to torch tensor
        X = torch.stack([x.to(dtype=torch.float32) for x in X])

        # Set the model to evaluation mode
        self.model.eval()

        # Forward pass
        outputs = self.forward(X)

        # Get the predicted class probabilities
        probabilities = nn.functional.softmax(outputs, dim=1)

        return probabilities

    def train(self, X, num_epochs=prm.cnn_epochs, batch_size=prm.cnn_batch_size, learning_rate=None):
        # Ensure that X is a list of torch tensors
        X = [x.to(dtype=torch.float32) for x in X]

        # Set the model to training mode
        self.model.train()

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate if learning_rate is None else learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            # Shuffle the data
            random.shuffle(X)

            # Split the data into batches
            num_batches = len(X) // batch_size
            for batch_idx in range(num_batches):
                # Get the current batch
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                batch_X = torch.stack(X[start_idx:end_idx])  # Stack the tensors into a single tensor

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_X)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

        return self
