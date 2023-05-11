import unittest
from torch import nn
from nn_executor import models as mm
from nn_executor.graph_analyzer import GraphDescription
from nn_executor.visualizer import GraphVisualDescription, draw_graph
import shared_tests_data

class TestVisualizer(unittest.TestCase):

    def test_drawing_repr(self):
        model = nn.Sequential(
                              mm.Identity(),
                              mm.Parallel(mm.Cat(1),
                                          nn.Sequential(
                                                        nn.Conv2d(5, 5, 1),
                                                        mm.ResBranch(5, 5, 1, 3),
                                                        mm.Parallel(mm.Cat(1),
                                                                    mm.ResBranch(5, 5, 1, 3),
                                                                    mm.ResBranch(5, 5, 1, 2)),
                                                        nn.Conv2d(10, 5, 1),
                                                        nn.Conv2d(5, 5, 1),
                                                        ),
                                          nn.Conv2d(5,1,1),
                                          mm.Identity(),
                                          ),
                              mm.Identity(),
                              )
        desc = shared_tests_data.get_model_description(model, [(1, 5, 128, 128)], True)
        gd = GraphDescription.make_from_model_description(desc)
        gvd = GraphVisualDescription.make_from_graph_description(gd)
        draw_graph(gvd, scale=(2,1))

if __name__ == "__main__":
    unittest.main()
