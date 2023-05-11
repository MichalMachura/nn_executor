from typing import Dict, List, Tuple
import unittest
from nn_executor import models as mm
from torch import abs_, nn

from nn_executor.static.analyzer import StaticAnalyzer
import shared_tests_data


class TestStaticAnalyzer(unittest.TestCase):

    def test_branch_detection(self):
        model = shared_tests_data.get_example_model_2()
        desc = shared_tests_data.get_model_description(model, [(1, 5, 128, 128)])
        analyzer = StaticAnalyzer(desc)
        branches = analyzer.analyze()

    # def test_branch_channels(self):
    #     model = shared_tests_data.ResNet(3, 13)
    #     desc = shared_tests_data.get_model_description(model, [(1, 3, 128, 128)])
    #     analyzer = StaticAnalyzer(desc)

if __name__ == "__main__":
    unittest.main()
