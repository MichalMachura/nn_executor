from unittest import TestCase
import unittest
import torch
from nn_executor import utils, executor
import shared_tests_data


class TestSaveLoad(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.pth = shared_tests_data.TEST_DIR + '/tmp/test_tmp.pth'
        self.json = shared_tests_data.TEST_DIR + '/tmp/test_tmp.json'
        self.desc_format_json = shared_tests_data.TEST_DIR + '/tmp/test_tmp_desc_format.json'
        self.desc_format_pth = shared_tests_data.TEST_DIR + '/tmp/test_tmp_desc_format.pth'

    def test_with_example_model_pth(self):

        model_description = shared_tests_data.get_example_description_1()
        # test save
        utils.save(self.pth, model_description)
        # test load
        md = utils.load(self.pth, torch.device('cpu'))

        with torch.no_grad():
            t1 = torch.rand((1, 3, 64, 64))
            t2 = torch.rand((1, 3, 3, 3))

            e1 = executor.Executor(model_description).eval()
            e2 = executor.Executor(md).eval()

            v1 = e1(t1, t2)
            v2 = e2(t1, t2)
            d1 = (v1[0] - v2[0]).abs().sum()
            d2 = (v1[1] - v2[1]).abs().sum()

            self.assertAlmostEqual(d1, 0.0, 3, "Results of models before and after save/load to file are different.")
            self.assertAlmostEqual(d2, 0.0, 3, "Results of models before and after save/load to file are different.")

    def test_with_example_model_json(self):

        model_description = shared_tests_data.get_example_description_1()
        # test save
        utils.save((self.json, self.pth), model_description)
        # test load
        md = utils.load((self.json, self.pth), torch.device('cpu'), True)

        with torch.no_grad():
            t1 = torch.rand((1, 3, 64, 64))
            t2 = torch.rand((1, 3, 3, 3))

            e1 = executor.Executor(model_description).eval()
            e2 = executor.Executor(md).eval()

            v1 = e1(t1, t2)
            v2 = e2(t1, t2)
            d1 = (v1[0] - v2[0]).abs().sum()
            d2 = (v1[1] - v2[1]).abs().sum()

            self.assertAlmostEqual(d1, 0.0, 3, "Results of models before and after save/load to file are different.")
            self.assertAlmostEqual(d2, 0.0, 3, "Results of models before and after save/load to file are different.")

    def test_layers_description_formats(self):

        model_description = shared_tests_data.get_example_description_1()
        # test load
        utils.save((self.desc_format_json, self.desc_format_pth), model_description)
        md = utils.load((self.desc_format_json, ), torch.device('cpu'), strict=False)

        with torch.no_grad():
            t1 = torch.rand((1, 3, 64, 64))
            t2 = torch.rand((1, 3, 3, 3))

            e1 = executor.Executor(model_description).eval()
            e2 = executor.Executor(md).eval()
            # use the same weights -- checked is only topology
            e2.load_state_dict(e1.state_dict())

            v1 = e1(t1, t2)
            v2 = e2(t1, t2)
            d1 = (v1[0] - v2[0]).abs().sum()
            d2 = (v1[1] - v2[1]).abs().sum()

            self.assertAlmostEqual(d1, 0.0, 3, "Results of models before and after save/load to file are different.")
            self.assertAlmostEqual(d2, 0.0, 3, "Results of models before and after save/load to file are different.")


if __name__ == "__main__":
    unittest.main()
