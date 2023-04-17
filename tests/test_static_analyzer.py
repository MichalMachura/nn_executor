import unittest


class TestStaticAnalyzer(unittest.TestCase):

    def test_branch_detection(self):
        import shared_tests_data
        from nn_executor.model_description import ModelDescription
        from nn_executor.static.analyzer import StaticAnalyzer

        import os, sys
        print(sys.path)

        desc = ModelDescription(shared_tests_data.get_model_description_1())
        analyzer = StaticAnalyzer()
        analyzer.analyze(desc)

        s = 0
        self.assertEqual(s, 0, "")
