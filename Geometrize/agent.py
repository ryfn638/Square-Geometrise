import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(120, 32, kernel_size=3, stride=1, padding=1),  # Extract edges
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Compress
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Compress more
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the output size dynamically
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.fc = nn.Linear(n_flatten, features_dim)  # Fully connected layer

    def forward(self, observations):
        return self.fc(self.cnn(observations))
