import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# Fourier Feature Layer
class FourierFeatureLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FourierFeatureLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(in_dim, out_dim) * 2 * torch.pi)
        
    def forward(self, x):
        return torch.cat([torch.sin(x @ self.weights), torch.cos(x @ self.weights)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim=512, output_activation=None):
        super(ResidualBlock, self).__init__()

        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        
        self.residual1 = nn.Linear(input_dim, 256) 
        self.residual2 = nn.Linear(256, 128)
        
        self.activation = nn.ReLU()  # Use ReLU for internal activations
        self.output_activation = output_activation  # This can be None, nn.Sigmoid(), nn.Tanh(), etc.
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        res1 = self.residual1(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = x + res1
        res2 = self.residual2(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = x + res2
        x = self.layer3(x)

        if self.output_activation:
            x = self.output_activation(x)
        
        return x

class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()
        
        self.conv1 = nn.Conv1d(k, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k*k)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        
        # Initialize the weights for the transformation to be close to the identity matrix
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        k = int((self.fc3.out_features)**0.5)  # Calculate the dimension size for reshaping
        x = self.fc3(x)
        return x.view(-1, k, k)

class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        
        # Input transform
        self.input_transform = TNet(k=6)
        self.conv1 = nn.Conv1d(6, 32, 1) 
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.feature_transform = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 512, 1)
        self.fc = nn.Linear(512, 256)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(512)
        
    def forward(self, x):
        matrix3x3 = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix3x3).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        matrix128x128 = self.feature_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix128x128).transpose(1, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]  # Global max pooling
        x = x.view(-1, 512)
        x = self.fc(x)
        
        return x
    
class SemanticNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=11):
        super(SemanticNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape the input for BatchNorm1d if necessary
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))  # Flatten batch and locations if needed

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
class ColorNet(nn.Module):
    def __init__(self, in_features=512, hidden_dim=256, num_bins=313):
        super(ColorNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)  # Use LayerNorm
        self.dropout1 = nn.Dropout(0.1) 

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_dim, 3 * num_bins)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])  # Flatten for LayerNorm
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.view(-1, 3, 313)
        x = F.softmax(x, dim=-1)  # Apply Softmax to color bins
        return x

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 3, 512) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return x
    

class TraversabilityNet(nn.Module):
    def __init__(self, input_dim=256):
        super(TraversabilityNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(-1, x.shape[-1])
        return self.mlp(x)



class MultiModalNetwork(nn.Module):
    def __init__(self, num_bins=313):
        super(MultiModalNetwork, self).__init__()
        self.pointnet_encoder = PointNetEncoder()
        self.audio_cnn = AudioCNN()
        self.fourier_layer = FourierFeatureLayer(3, 256)
        self.compression_layer = nn.Linear(1280, 512)
        self.sdf_fcn = ResidualBlock(input_dim=512, output_activation=nn.Tanh())
        self.confidence_fcn = ResidualBlock(input_dim=512, output_activation=nn.Sigmoid())
        self.semantic_fcn = SemanticNet(input_dim=512)
        self.color_fcn = ColorNet(in_features=512, hidden_dim=128, num_bins=num_bins)
        self.traversability_fc = TraversabilityNet(input_dim=512)

    def forward(self, locations, point_clouds, audio, scan_indices=None):
        batch_size, num_locations, _ = locations.shape

        if scan_indices is not None:
            unique_scans, inverse_indices = torch.unique(scan_indices, return_inverse=True)
            unique_pc_features = self.pointnet_encoder(point_clouds[unique_scans].transpose(-1, -2))
            unique_audio_features = self.audio_cnn(audio[unique_scans])
            pc_features = unique_pc_features[inverse_indices]
            audio_features = unique_audio_features[inverse_indices]
        else:
            pc_features = self.pointnet_encoder(point_clouds.transpose(-1, -2))
            audio_features = self.audio_cnn(audio)

        location_features = self.fourier_layer(locations.view(-1, 3))

        num_queries = location_features.shape[0]
        pc_features = pc_features.repeat_interleave(num_locations // pc_features.shape[0], dim=0)
        audio_features = audio_features.repeat_interleave(num_locations // audio_features.shape[0], dim=0)
        distance_weights = self.compute_distance_weight(locations)
        weighted_pc_features = pc_features * distance_weights
        weighted_audio_features = audio_features * distance_weights

        if location_features.dim() == 2:
            location_features = location_features.unsqueeze(0)  # Adds batch dimension [1, 4096, 512]

        assert location_features.shape[1] == weighted_pc_features.shape[1] == weighted_audio_features.shape[1], "Query points mismatch"

        combining_features = torch.cat([location_features, weighted_pc_features, weighted_audio_features], dim=2)
        compressed_features = self.compression_layer(combining_features)

        # Generate predictions
        sdf = self.sdf_fcn(compressed_features).view(batch_size, num_locations)
        confidence = self.confidence_fcn(compressed_features).view(batch_size, num_locations)
        semantics = self.semantic_fcn(compressed_features)
        color_logits = self.color_fcn(compressed_features)
        traversability = self.traversability_fc(compressed_features).view(batch_size, num_locations)

        return sdf, confidence, semantics, color_logits, traversability

    def compute_distance_weight(self, locations, sigma=1):
        origin = torch.zeros_like(locations)
        distances = torch.norm(locations - origin, dim=-1, keepdim=True)
        weights = torch.exp(- distances / sigma)
        weights = weights / torch.max(weights)
        
        return weights
