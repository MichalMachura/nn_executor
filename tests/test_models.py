import unittest
import torch
from nn_executor import models


class TestVariable(unittest.TestCase):

    def test_create(self,):
        v = models.Variable(t=torch.empty((1, 31, 40, 40), dtype=torch.float32))
