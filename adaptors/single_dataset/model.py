import torch
import torch.nn as nn


class MitoModel(nn.Module):
    def __init__(self, input_dim, num_classes=None, task='classification'):
        """
        Args:
            input_dim (int): The dimensionality of the input embeddings.
            num_classes (int, optional): The number of classes for classification.
                Must be provided if task is 'classification'.
            task (str): Either 'classification' or 'regression'.
                For classification, the output dimension is num_classes.
                For regression, the output dimension is 1.
        """
        super(MitoModel, self).__init__()
        self.task = task.lower()

        if self.task == 'classification':
            if num_classes is None:
                raise ValueError("For classification tasks, you must provide a valid num_classes integer.")
            self.fc2 = nn.Linear(input_dim, num_classes)
        elif self.task == 'regression':
            self.regressor = nn.Sequential(nn.Linear(input_dim, input_dim//2),
                                           nn.ReLU(),
                                           nn.Linear(input_dim//2, 1))
        else:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    def forward(self, x):
        """
        Forward pass: apply a linear layer, ReLU activation, and another linear layer.
        """
        if self.task == 'classification':
            return self.fc2(x)
        else:
            return self.regressor(x)


class MitoTemporalModel(nn.Module):
    def __init__(self, input_dim, num_classes=None, task='regression'):
        # the embeddings are of shape (N, t, d)
        # the model is 1d conv model

        super(MitoTemporalModel, self).__init__()

        self.regressor = nn.Sequential(
            nn.Conv1d(input_dim, input_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(input_dim//2, input_dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(input_dim//4, input_dim//8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(input_dim//8, input_dim//16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_dim//16, input_dim//32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_dim//32, 1, kernel_size=3, padding=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.regressor(x)
        x = self.avg_pool(x)

        return x.squeeze(2)



# --- Example usage ---
if __name__ == "__main__":
    input_dim = 2048  # Example embedding size
    dummy_input = torch.randn(10, input_dim)  # 10 sample embeddings

    # Classification example:
    num_classes = 27
    model_classification = MitoModel(input_dim, num_classes=num_classes, task='classification')
    print(model_classification)
    output_classification = model_classification(dummy_input)
    print("Classification output shape:", output_classification.shape)  # Expected: [10, 27]

    # Regression example:
    model_regression = MitoModel(input_dim, task='regression')
    print(model_regression)
    output_regression = model_regression(dummy_input)
    print("Regression output shape:", output_regression.shape)  # Expected: [10, 1]