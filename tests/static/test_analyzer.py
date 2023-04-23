import unittest

from nn_executor.model_description import ModelDescription
from nn_executor.static.analyzer import StaticAnalyzer

import os, sys
__path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if __path not in sys.path:
    sys.path.append(__path)
import shared_tests_data


class TestStaticAnalyzer(unittest.TestCase):

    def test_branch_detection(self):
        import os, sys

        desc = ModelDescription(shared_tests_data.get_model_description_1())

        analyzer = StaticAnalyzer()

        branches = analyzer.get_branches(desc)
        print(branches)
        pass

if __name__ == "__main__":
    TestStaticAnalyzer().test_branch_detection()
    # unittest.main()