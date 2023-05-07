from typing import Dict, List, Tuple
import unittest
from nn_executor import models as mm
from torch import abs_, nn

from nn_executor.static.analyzer import StaticAnalyzer
import shared_tests_data


class TestStaticAnalyzer(unittest.TestCase):

    def test_branch_detection(self):
        desc = shared_tests_data.get_example_description_1()

        analyzer = StaticAnalyzer()
        branches = analyzer.get_branches(desc)
        print(branches)

    def test_branch_channels(self):
        model = shared_tests_data.ResNet(3, 13)
        desc = shared_tests_data.get_model_description(model, [(1, 3, 128, 128)])
        analyzer = StaticAnalyzer(desc)
        branches = analyzer.get_branches(desc)
        branches


if __name__ == "__main__":
    unittest.main()
