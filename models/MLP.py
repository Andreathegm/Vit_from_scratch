import torch.nn as nn
import torch

class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, representation_size= None):
        super().__init__()
        self.representation_size = representation_size

        if representation_size is None:
            self.pre_logits = nn.Identity()
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.pre_logits = nn.Linear(embed_dim, representation_size)
            self.act = nn.Tanh()
            self.head = nn.Linear(representation_size, num_classes)

    def forward(self, x):
        x = self.pre_logits(x)
        if self.representation_size is not None:
            x = self.act(x)
        x = self.head(x)
        return x


head = MLPHead(embed_dim=768, num_classes=10, representation_size=None)
cls = torch.randn(64, 768)
with torch.no_grad():
    logits = head(cls)
print("MLPHead logits:", logits.shape)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

mlp = MLP(embed_dim=768, mlp_ratio=4.0, dropout=0.0)
x = torch.randn(64, 197, 768)
with torch.inference_mode():
    y = mlp(x)
print("MLP out:", y.shape)