import torch


class E_ReLU(torch.nn.Module):

    def __init__(self, e=1e-6, inplace: bool = False):
        super().__init__()
        self.e = e
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(input, inplace=self.inplace) + self.e
