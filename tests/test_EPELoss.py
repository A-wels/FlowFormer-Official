import pytest
import torch
import sys
import os

sys.path.append(os.getcwd())
from myutils.EPELoss import EPELoss

# class for testing nn.Module EPELoss
class TestEPELoss:
    criterion = EPELoss()

    def test_one(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
        y = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
        loss = self.criterion(x, y)
        assert loss == 0.0

    def test_two(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
        y = torch.tensor([[[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]])
        loss = self.criterion(x, y)
        assert loss == 1.0

    def test_three(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
        y = torch.tensor([[[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]]])
        loss = self.criterion(x, y)
        assert loss == 2.0
        
    def test_zeroes(self):
        x = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
        y = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
        loss = self.criterion(x, y)
        assert loss == 0.0

