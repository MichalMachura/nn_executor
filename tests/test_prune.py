from typing import List, Tuple, Union
import unittest
import torch
from nn_executor import modifiers, models, parser, utils
from nn_executor.executor import Executor
from nn_executor.model_description import ModelDescription
from nn_executor.pruning import prune
import shared_tests_data


def append_conv(model: torch.nn.Module,
                ch_in: int,
                ch_out: int = 1):
    conv = torch.nn.Conv2d(ch_in, ch_out, 3, padding=1)
    return torch.nn.Sequential(model, conv)


def pre_conv(model: torch.nn.Module,
             ch_in: int,
             ch_out: int):
    conv = torch.nn.Conv2d(ch_in, ch_out, 3, padding=1)
    return torch.nn.Sequential(conv, model)


class TestScissors(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.modifiers_map = {
            torch.nn.Conv2d: modifiers.Conv2dModifier(),
            torch.nn.MaxPool2d: modifiers.MaxPool2dModifier(),
            torch.nn.ReLU: modifiers.ReLUModifier(),
            torch.nn.BatchNorm2d: modifiers.BatchNorm2dModifier(),
            models.Pruner: modifiers.PrunerModifier(True),
            models.Identity: modifiers.IdentityModifier(),
            models.Cat: modifiers.CatModifier(),
            models.Add: modifiers.AddModifier(),
            models.Upsample: modifiers.UpsampleModifier(),
        }
        self.pth = shared_tests_data.TEST_DIR + '/tmp/test_prune_tmp_{}_before.pth'
        self.json = shared_tests_data.TEST_DIR + '/tmp/test_prune_tmp_{}_after.json'

        self.chain_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(5, 3, 3, padding=1),
                                shared_tests_data.pruner([1, 1, 0]),
                                torch.nn.Conv2d(3, 5, 3, padding=1),
                                shared_tests_data.pruner([1, 1, 0, 1, 0], False),
                            )
        self.chain_1_ref = torch.nn.Sequential(
                                torch.nn.Conv2d(5, 2, 3, padding=1),
                                models.Identity(),
                                torch.nn.Conv2d(2, 5, 3, padding=1),
                                models.Identity(),
                            )

        self.ADD_1 = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            self.chain_1,
                            ])
        self.ADD_1_ref = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            self.chain_1_ref
                            ])
        ############################################
        self.ADD_2 = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 3, 3, padding=1),
                                shared_tests_data.pruner([0, 0, 0]),
                                torch.nn.Conv2d(3, 5, 3, padding=1),
                                shared_tests_data.pruner([1, 0, 0, 1, 0], False),
                            )
                            ])
        self.ADD_2_ref = models.Parallel(models.Add(2),
                            [
                            models.Identity()
                            ])
        ############################################
        self.ADD_3 = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 8, 3, padding=1),
                                shared_tests_data.pruner([1, 1, 1, 0, 0, 0, 1, 1]),
                                torch.nn.Conv2d(8, 5, 3, padding=1),
                                shared_tests_data.pruner([0, 1, 0, 1, 0], False),
                            )
                            ])
        self.ADD_3_ref = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 5, 3, padding=1),
                                models.Identity(),
                                torch.nn.Conv2d(5, 5, 3, padding=1),
                                models.Identity(),
                            )
                            ])
        ############################################

        self.R1 = torch.nn.Sequential(
            torch.nn.Conv2d(5, 5, 3, padding=1),
            shared_tests_data.pruner([1, 1, 1, 1, 0]),
            self.ADD_1,
            shared_tests_data.pruner([1, 1, 1, 1, 0]),
            self.ADD_2,
            shared_tests_data.pruner([1, 1, 1, 1, 0]),
            self.ADD_3,
            shared_tests_data.pruner([1, 1, 1, 1, 0]),
            torch.nn.Conv2d(5, 5, 3, padding=1),
        )
        self.R1_ref = torch.nn.Sequential(
            torch.nn.Conv2d(5, 4, 3, padding=1),
            # shared_tests_data.pruner([1, 1, 1, 1, 0]),
            models.Identity(),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(4, 2, 3, padding=1),
                                models.Identity(),
                                torch.nn.Conv2d(2, 4, 3, padding=1),
                                models.Identity(),
                            )
                            ]),
            # shared_tests_data.pruner([1, 1, 1, 1, 0]),
            models.Identity(),
            models.Parallel(models.Add(2),
                            [
                            models.Identity()
                            ]),
            # shared_tests_data.pruner([1, 1, 1, 1, 0]),
            models.Identity(),
            models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(4, 5, 3, padding=1),
                                models.Identity(),
                                torch.nn.Conv2d(5, 4, 3, padding=1),
                                models.Identity(),
                            )
                            ]),
            # shared_tests_data.pruner([1, 1, 1, 1, 0]),
            models.Identity(),
            torch.nn.Conv2d(4, 5, 3, padding=1),
        )
        ############################################

        self.ADD_4 = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 3, 3, padding=1),
                                shared_tests_data.pruner([1, 1, 0]),
                                torch.nn.Conv2d(3, 5, 3, padding=1),
                                shared_tests_data.pruner([1, 1, 0, 1, 0], False),
                            )
                            ])
        self.ADD_4_ref = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 2, 3, padding=1),
                                models.Identity(),
                                torch.nn.Conv2d(2, 5, 3, padding=1),
                                models.Identity(),
                            )
                            ])
        ############################################
        self.ADD_5 = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 3, 3, padding=1),
                                shared_tests_data.pruner([0, 0, 0]),
                                torch.nn.Conv2d(3, 5, 3, padding=1),
                                shared_tests_data.pruner([1, 0, 0, 1, 0], False),
                            )
                            ])
        self.ADD_5_ref = models.Parallel(models.Add(2),
                            [
                            models.Identity()
                            ])
        ############################################
        self.ADD_6 = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 8, 3, padding=1),
                                shared_tests_data.pruner([1, 1, 1, 0, 0, 0, 1, 1]),
                                torch.nn.Conv2d(8, 5, 3, padding=1),
                                shared_tests_data.pruner([0, 1, 0, 1, 0], False),
                            )
                            ])
        self.ADD_6_ref = models.Parallel(models.Add(2),
                            [
                            models.Identity(),
                            torch.nn.Sequential(
                                torch.nn.Conv2d(5, 5, 3, padding=1),
                                models.Identity(),
                                torch.nn.Conv2d(5, 5, 3, padding=1),
                                models.Identity(),
                            )
                            ])
        ############################################
        self.R2 = torch.nn.Sequential(
            shared_tests_data.pruner([0, 0, 0, 0, 0]),
            self.ADD_4,
            shared_tests_data.pruner([0, 0, 0, 0, 0]),
            self.ADD_5,
            shared_tests_data.pruner([0, 0, 0, 0, 0]),
            self.ADD_6,
            shared_tests_data.pruner([0, 0, 0, 0, 0]),
        )
        self.R2_ref = models.Absorber()

        ############################################
        self.BRANCH_1 = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 6, 3, padding=1),
                    shared_tests_data.pruner([1, 1, 1, 0, 0, 0]),
                    torch.nn.Conv2d(6, 6, 3, padding=1),
                    shared_tests_data.pruner([1, 1, 0, 0, 1, 0]),
                    shared_tests_data.pruner([1, 1, 1, 0, 1, 0]),
                    torch.nn.Conv2d(6, 2, 3, padding=1),
                    shared_tests_data.pruner([1, 0]),
                )
        self.BRANCH_1_ref = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 3, 3, padding=1),
                    models.Identity(),
                    torch.nn.Conv2d(3, 3, 3, padding=1),
                    models.Identity(),
                    models.Identity(),
                    torch.nn.Conv2d(3, 1, 3, padding=1),
                    models.Identity(),
                )
        ############################################
        self.BRANCH_2 = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 6, 9, padding=4),
                    shared_tests_data.pruner([0, 0, 0, 0, 0, 0]),
                    torch.nn.Conv2d(6, 2, 3, padding=1),
                )
        self.BRANCH_2_ref = models.Absorber()
        ############################################
        self.BRANCH_3 = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 6, 3, padding=1),
                    shared_tests_data.pruner([1, 1, 1, 1, 1, 1]),
                    torch.nn.Conv2d(6, 5, 3, padding=1),
                    shared_tests_data.pruner([1, 0, 1, 0, 1]),
                )
        self.BRANCH_3_ref = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 6, 3, padding=1),
                    models.Identity(),
                    torch.nn.Conv2d(6, 3, 3, padding=1),
                    models.Identity(),
                )
        ############################################
        self.BRANCH_4 = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 6, 3, padding=1),
                    shared_tests_data.pruner([1, 1, 0, 1, 1, 1]),
                    torch.nn.Conv2d(6, 3, 3, padding=1),
                    shared_tests_data.pruner([1, 0, 0]),
                )
        self.BRANCH_4_ref = torch.nn.Sequential(
                    torch.nn.Conv2d(5, 5, 3, padding=1),
                    models.Identity(),
                    torch.nn.Conv2d(5, 1, 3, padding=1),
                    models.Identity(),
                )
        ############################################

        self.CAT_1 = models.Parallel(models.Cat(1),
                            [
                models.Identity(),
                self.BRANCH_1,
                self.BRANCH_2,
                self.BRANCH_3,
                self.BRANCH_4,
            ])
        self.CAT_1_ref = models.Parallel(models.Cat(1),
                            [
                models.Identity(),
                self.BRANCH_1_ref,
                self.BRANCH_2_ref,
                self.BRANCH_3_ref,
                self.BRANCH_4_ref,
            ])
        ############################################
        self.CAT_2 = models.Parallel(models.Cat(1), [self.R1, self.R2])
        self.CAT_2_ref = models.Parallel(models.Cat(1), [self.R1_ref, self.R2_ref])
        ############################################

        L = [
            torch.nn.Conv2d(3, 5, 3, padding=1),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
            shared_tests_data.pruner([1, 0, 1, 1, 0]),
            torch.nn.MaxPool2d(2, 2),
            models.Upsample(size=(160, 80), mode='nearest'),
            torch.nn.MaxPool2d(2, 2),
            self.CAT_1,

            shared_tests_data.pruner([1, 1, 1, 1, 0,
                           0, 1,
                           1, 1,
                           1, 1, 1, 0, 1,
                           0, 1, 0]),
            torch.nn.Conv2d(17, 8, 7, padding=3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            shared_tests_data.pruner([1, 1, 1, 1, 0, 0, 0, 0]),

            torch.nn.Conv2d(8, 5, 3, padding=1),
            self.CAT_2,
            torch.nn.Conv2d(10, 5, 3, padding=1),
            torch.nn.Conv2d(5, 2, 3, padding=1),
            torch.nn.Conv2d(2, 4, 3, padding=1),
            shared_tests_data.pruner([1, 1, 0, 1], False, True),
        ]
        self.model = torch.nn.Sequential(*L)
        ############################################

    def __make_description(self,
                           model: torch.nn.Module,
                           input_shape: Tuple[int],
                           differentiate_tensors: bool) -> ModelDescription:
        with utils.DifferentiateTensors(differentiate_tensors):
            p = parser.Parser()
            t = torch.rand(input_shape, dtype=torch.float32)
            md = p.parse_module(model, t)
        return md

    def __reparse_model_description(self,
                                    model_description: ModelDescription,
                                    input_shape: Tuple[int, int, int, int],
                                    differentiate_tensors: bool) -> ModelDescription:
        model = Executor(model_description)
        md = self.__make_description(model, input_shape, differentiate_tensors)

        # simulate writing to file
        storage_format = utils.model_description_to_storage_format(md)
        md = utils.storage_format_to_model_description(storage_format)

        return md

    def __compare_layer_indices(self, base_indices: List[int], ref_indices: List[int]):
        self.assertEqual(len(base_indices), len(ref_indices),
                         f"Layers indices have different lengths: {len(base_indices)}, {len(ref_indices)}")

        for i, (base, ref) in enumerate(zip(base_indices, ref_indices)):
            self.assertEqual(base, ref, f"Different indices at pos {i}")

    def __compare_unique_layers(self,
                                base_layers: List[torch.nn.Module],
                                ref_layers: List[torch.nn.Module],
                                ):
        self.assertEqual(len(base_layers), len(ref_layers),
                         f"Layers lists have different lengths: {len(base_layers)}, {len(ref_layers)}")

        for i, (base, ref) in enumerate(zip(base_layers, ref_layers)):
            self.assertTrue(isinstance(base, ref.__class__) and isinstance(ref, base.__class__),
                            f"Different layers classes at pos {i}")

    def __compare_model_descriptions(self,
                                     model_base: ModelDescription,
                                     model_ref: ModelDescription,
                                     input_shape: Tuple[int, int, int, int],
                                     differentiate_tensors: bool
                                     ):
        model_base = self.__reparse_model_description(model_base, input_shape, differentiate_tensors)
        model_ref = self.__reparse_model_description(model_ref, input_shape, differentiate_tensors)

        self.__compare_layer_indices(model_base.layers_indices, model_ref.layers_indices)
        self.__compare_unique_layers(model_base.unique_layers, model_ref.unique_layers)

    def _prune(self,
               model_description: ModelDescription,
               input_shape = (1, 5, 80, 40),
               differentiate_tensors: bool = False,
               save_name: str = None) -> ModelDescription:
        with utils.DifferentiateTensors(differentiate_tensors):
            t = torch.rand(input_shape, dtype=torch.float32)
            sc = prune.Scissors(model_description, self.modifiers_map)
            model_description = sc(t, parser.SUPPORTED_MODULES)

        if save_name is not None:
            utils.save((self.json.format(save_name),
                        self.pth.format(save_name)), model_description)

        return model_description

    def __prune_and_compare(self,
                            model_description: torch.nn.Module,
                            ref_description: torch.nn.Module,
                            input_shape,
                            differentiate_tensors: bool,
                            save_name: str = None
                            ):
        input_shape = (1, 5, 80, 40)
        model_description = self.__make_description(model_description, input_shape, differentiate_tensors)
        ref_description = self.__make_description(ref_description, input_shape, differentiate_tensors)
        pruned_desc = self._prune(model_description,
                                  input_shape=input_shape,
                                  differentiate_tensors=differentiate_tensors,
                                  save_name=save_name)

        self.__compare_model_descriptions(pruned_desc, ref_description, input_shape, differentiate_tensors)

    def test_prune_chain_1(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.chain_1, self.chain_1_ref, input_shape, differentiate_tensors=True, save_name='chain_1')

    def test_prune_add_1(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.ADD_1, self.ADD_1_ref, input_shape, differentiate_tensors=True, save_name='ADD_1')

    def test_prune_add_2(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.ADD_2, self.ADD_2_ref, input_shape, differentiate_tensors=True, save_name='ADD_2')

    def test_prune_add_3(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.ADD_3, self.ADD_3_ref, input_shape, differentiate_tensors=True, save_name='ADD_3')

    def test_prune_R1(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.R1, self.R1_ref, input_shape, differentiate_tensors=True, save_name='R1')

    def test_prune_add_4(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.ADD_4, self.ADD_4_ref, input_shape, differentiate_tensors=True, save_name='ADD_4')

    def test_prune_add_5(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.ADD_5, self.ADD_5_ref, input_shape, differentiate_tensors=True, save_name='ADD_5')

    def test_prune_add_6(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.ADD_6, self.ADD_6_ref, input_shape, differentiate_tensors=True, save_name='ADD_6')

    def test_prune_R2(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.R2, self.R2_ref, input_shape, differentiate_tensors=True, save_name='R2')

    def test_prune_branch_1(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(append_conv(self.BRANCH_1, 2, 1),
                                 append_conv(self.BRANCH_1_ref, 1, 1),
                                 input_shape, differentiate_tensors=True,
                                 save_name='BRANCH_1')

    def test_prune_branch_2(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(self.BRANCH_2, self.BRANCH_2_ref, input_shape, differentiate_tensors=True, save_name='BRANCH_2')

    def test_prune_branch_3(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(append_conv(self.BRANCH_3, 5 ,1),
                                 append_conv(self.BRANCH_3_ref, 3 ,1),
                                 input_shape, differentiate_tensors=True,
                                 save_name='BRANCH_3')

    def test_prune_branch_4(self):
        input_shape = (1, 5, 80, 40)
        self.__prune_and_compare(append_conv(self.BRANCH_4, 3, 1),
                                 append_conv(self.BRANCH_4_ref, 1, 1),
                                 input_shape, differentiate_tensors=True,
                                 save_name='BRANCH_4')


if __name__ == '__main__':
    unittest.main()
