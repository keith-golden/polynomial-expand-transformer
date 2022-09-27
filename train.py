import argparse
import os
import re
from timeit import default_timer as timer
from typing import List
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from model import Seq2SeqTransformer, create_mask, generate_square_subsequent_mask
from utils import PolynomialDataset, yield_tokens


SRC_LANGUAGE = 'factors'
TGT_LANGUAGE = 'expansions'

text_transform = {}  # maps SRC_LANGUAGE or TGT_LANGUAGE to sequential transforms to perform on text data
vocab_transform = {}  # maps SRC_LANGUAGE or TGT_LANGUAGE to torch Vocab objects

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Get path to current working directory
file_dir = os.getcwd()
# specify data directory
data_dir = file_dir + "/data"



#######################################
# COLLATION FUNCTIONS
# Functions below convert string pairs into a batch of tensors that can
# be fed into our model
#######################################

def sequential_transforms(*transforms):
    """ A helper function to club together sequential operations.
    """
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    """ This takes a list of token ids, adds the BOS and EOS tokens to the list of token ids,
    and converts the entire list into a tensor.
    """
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def token_transform(polynomial: str):
    """ Tokenizes a single string polynomial (think factor OR expansion) into a
    list of string tokens

    :param polynomial: a single string polynomial
    :return polynomial_list: a list of string tokens
    """
    # obtain all individual symbols/numbers/words into a list
    polynomial_list = re.findall(r"cos|sin|tan|\d|\w|\(|\)|\+|-|\*+", polynomial.strip().lower())
    return list(polynomial_list)


def collate_fn(batch):
    """Convert batch of raw strings into a batch of tensors."""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def build_preprocessing_objects(train_iter: PolynomialDataset):
    """Builds the Vocab objects for source and target languages from the training set,
    and the sequential transformation objects used to preprocess the training data.

    :param: train_iter: PolynomialDataset object for training data

    :returns: vocab_transform: dict mapping SRC_LANGUAGE or TGT_LANGUAGE to torch Vocab objects
    :returns: text_transform: dict mapping SRC_LANGUAGE or TGT_LANGUAGE to sequential transforms
    to perform on text data
    """

    # Build torchtext Vocab object for factors (inputs)
    vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_iter.factors),
                                                              min_freq=1,
                                                              specials=special_symbols,
                                                              special_first=True)

    # Build torchtext Vocab object for expansions (target)
    vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_iter.expansions),
                                                              min_freq=1,
                                                              specials=special_symbols,
                                                              special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    # src and tgt language text transforms to convert raw strings into tensors indices
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform,  # Tokenize inputs
                                                   vocab_transform[ln],  # Numericalize the tokens
                                                   tensor_transform)  # Add BOS/EOS and create tensor

    return vocab_transform, text_transform


#######################################
# TRAINING/EVALUATION FUNCTIONS
# Functions below are used during training phase
#######################################

def train_epoch(model, optimizer, train_iterable):
    """Train the model for a single epoch."""
    model.train()
    losses = 0

    train_dataloader = DataLoader(train_iterable, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    batch_count = 1
    len_train_dataloader = train_dataloader.__len__()

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        if batch_count % 100 == 0:
            print("Epoch % complete:", batch_count / len_train_dataloader * 100)
        batch_count += 1

    return losses / len(train_dataloader)


def evaluate(model, val_file):
    """Evaluate the current loss of the model on the validation set."""
    model.eval()
    losses = 0

    val_iter = PolynomialDataset(val_file, data_dir)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


#######################################
# PREDICTION FUNCTIONS
# Functions below are used to make model predictions
#######################################

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """Generate output sequence using greedy algorithm"""
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model: torch.nn.Module, src_sentence: str):
    """Translates input sentence (factor) into target language (expansion).

    :return: model's predicted translation of input
    """
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return "".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="train_cropped.txt")
    parser.add_argument("--validation_file", type=str, default="validation.txt")
    parser.add_argument("--num_epochs", type=int, default=10)  # Number of epochs to train the model
    parser.add_argument("--weight_file", type=str)  # Loads model and optimizer weights at the specified file
    parser.add_argument("--save_epoch_weights", action='store_true')  # Save model weights at each epoch
    args = parser.parse_args()

    # Build dataset object for training dataset
    train_file = args.train_file
    train_iter = PolynomialDataset(train_file, data_dir)

    # Build source and target language vocab objects, collate their preprocessing functions
    # and store them in dictionaries
    vocab_transform, text_transform = build_preprocessing_objects(train_iter)

    # Set model seed and define its parameters
    torch.manual_seed(0)
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1

    # Instantiate model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    # Perform xavier initialization of parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Instantiate model, loss function, and optimizer
    transformer = transformer.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # load previous optimizer and model weights
    if args.weight_file:
        weight_file_name = args.weight_file
        weights = torch.load(weight_file_name, map_location=DEVICE)
        transformer.load_state_dict(weights['transformer_state_dict'])
        optimizer.load_state_dict(weights['transformer_optimizer'])

    # print number of parameters
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("# of parameters:", num_params)

    # Now we have all the ingredients to train our model
    NUM_EPOCHS = args.num_epochs

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_iter)
        end_time = timer()
        val_loss = evaluate(transformer, "validation.txt")
        print((
                  f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = "
                  f"{(end_time - start_time):.3f}s"))

        # Save the model weights at this epoch if specified
        if args.save_epoch_weights:
            file_name = file_dir + "/Transformer_Weights_Epoch_" + str(epoch) + ".pt"
            torch.save({
                'transformer_state_dict': transformer.state_dict(),
                'transformer_optimizer': optimizer.state_dict(),
            }, file_name)

