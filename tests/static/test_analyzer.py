import unittest

from nn_executor.model_description import ModelDescription
from nn_executor.static.analyzer import StaticAnalyzer

import os, sys
__path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(__path)
if __path not in sys.path:
    sys.path.append(__path)
import shared_tests_data


class TestStaticAnalyzer(unittest.TestSuite):

    def test_branch_detection(self):
        import os, sys
        print(sys.path)

        desc = ModelDescription(shared_tests_data.get_model_description_1())

        analyzer = StaticAnalyzer()

        analyzer.analyze(desc)
        s = 0
        pass
