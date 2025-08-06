import torch
import torch.nn as nn
import torch.nn.init as init


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward(self, x):
        identity = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        out = self.bn2(out)

        out = out + identity
        
        out = self.activation(out)
        
        return out


class ResidualModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=5, num_blocks=3):
        super(ResidualModel, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.dropout = nn.Dropout(0.2)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, ResidualBlock):
                init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward(self, x):
        epsilon = 1e-8
        x = x + epsilon
        
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.activation(x)
        
        for block in self.residual_blocks:
            x = block(x)
            x = self.dropout(x)
        
        x = self.output_layer(x)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x)
        
        return x


def get_model(model_type, input_dim, hidden_dim=64, output_dim=5):
    actual_input_dim = input_dim + 15
    
    if model_type == 'residual':
        return ResidualModel(actual_input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


