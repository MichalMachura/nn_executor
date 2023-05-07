import unittest
from torch import nn
from nn_executor.graph_analyzer import GraphDescription
from nn_executor import models as mm
import shared_tests_data


class TestGraphAnalyzer(unittest.TestCase):

    def test_of_model_description_parsing(self):
        nn.Identity
        model = nn.Sequential(
                              mm.Identity(),
                              mm.Parallel(mm.Cat(1),
                                          nn.Sequential(
                                                        nn.Conv2d(5, 5, 1),
                                                        # mm.ResBranch(5, 5, 1, 2),
                                                        # mm.Parallel(mm.Cat(1),
                                                        #             mm.ResBranch(5, 5, 1, 3),
                                                        #             mm.ResBranch(5, 5, 1, 2)),
                                                        # nn.Conv2d(10, 5, 1),
                                                        nn.Conv2d(5, 5, 1),
                                                        ),
                                          nn.Conv2d(5,1,1),
                                          mm.Identity(),
                                          ),
                              mm.Identity(),

                              )
        desc = shared_tests_data.get_model_description(model, [(1, 5, 128, 128)], True)
        graph_description = GraphDescription.make_from_model_description(desc)


if __name__ == "__main__":
    unittest.main()
