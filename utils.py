import re
from typing import Iterable, List, Tuple
import torch


# Inherits Pytorch Dataset object
class PolynomialDataset(torch.utils.data.Dataset):

    def __init__(self, data_file, file_dir, transform=None, target_transform=None):

        # load file using function from main
        full_path = file_dir + "/" + data_file
        factors, expansions = load_file(full_path)

        self.factors = factors
        self.expansions = expansions
        self.file_dir = file_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx):

        factor = self.factors[idx]
        expansion = self.expansions[idx]

        if self.transform:
            factor = self.transform(factor)
        if self.target_transform:
            expansion = self.target_transform(expansion)
        return factor, expansion


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def yield_tokens(data_iter: Iterable) -> List[str]:
    """Helper function to yield a list of tokens from a sample."""

    for polynomial_sample in data_iter:
        # return a list of all tokens in this sample
        polynomial_list = list(re.findall(r"cos|sin|tan|\d|\w|\(|\)|\+|-|\*+", polynomial_sample.strip().lower()))
        yield polynomial_list
