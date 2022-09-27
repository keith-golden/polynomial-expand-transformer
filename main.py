import pickle
import sys
import numpy as np
from typing import Tuple
import torch
import train
from model import Seq2SeqTransformer
import os

from utils import PolynomialDataset

global input_lang
global output_lang
MAX_SEQUENCE_LENGTH = 29
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factor: str):

    pred_factor = train.translate(transformer, factor)

    return pred_factor


# --------- END OF IMPLEMENT THIS --------- #


def main(filename: str):
    # Go to data directory and get path to this file
    filepath = os.getcwd() + "/data/" + filename

    factors, expansions = load_file(filepath)
    preds = [predict(f) for f in factors]

    # Print out predictions and true labels
    for p, e, in zip(preds, expansions):
        print(p, ":", e)

    scores = [score(te, pe) for te, pe in zip(expansions, preds)]
    print(np.mean(scores))


if __name__ == "__main__":

    # Rebuild Vocab and preprocessing objects from training data, stored as dicts
    # Consider loading these as pickled objects
    train_iter = PolynomialDataset("train_cropped.txt", os.getcwd() + "/data/")
    vocab_transform, text_transform = train.build_preprocessing_objects(train_iter)

    # Define model params and instantiate model
    SRC_VOCAB_SIZE = len(train.vocab_transform[train.SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(train.vocab_transform[train.TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    # Load model weights
    PATH = "Transformer_Weights.pt"
    weights = torch.load(PATH, map_location=DEVICE)

    transformer.load_state_dict(weights['transformer_state_dict'])
    transformer.eval()

    # NOTE: test.txt SHOULD RESIDE IN THE DATA FOLDER!!
    main("test.txt" if "-t" in sys.argv else "train.txt")
