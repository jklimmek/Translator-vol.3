import argparse
import json
import os
from collections import Counter

import numpy as np
from tqdm import tqdm

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .utils import read_txt


def train_tokenizer(
        text, 
        vocab_size,
        pad_token,
        unk_token,
        start_token,
        end_token,
        additional_alphabet=list()
    ):
    """
    Trains a WordPiece tokenizer on a list of text sequences.

    Parameters:
        text (list): List of text sequences.
        vocab_size (int): Desired vocabulary size.
        pad_token (str): Padding token.
        unk_token (str): Unknown token.
        start_token (str): Start of sequence token.
        end_token (str): End of sequence token.
        additional_alphabet (list): Additional alphabet to use.

    Returns:
        tokenizer (Tokenizer): The trained tokenizer.
    """

    # Initial alphabet.
    alphabet = list(
        "!\"#$£%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`\
        abcdefghijklmnopqrstuvwxyz{|}~ÀÂÇÉÈÊËÎÏÔÙÛÜàâçéèêëîïôùûü«»")

    # Extend alphabet with additional symbols.
    alphabet.extend(additional_alphabet)

    # Initialize tokenizer.
    tokenizer = Tokenizer(models.WordPiece(unk_token=unk_token))

    # Pre-tokenization.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Punctuation(),
        pre_tokenizers.Whitespace()
    ])

    # Special tokens.
    special_tokens = [
        pad_token, 
        unk_token,
        start_token, 
        end_token
    ]

    # Train tokenizer.
    trainer = trainers.WordPieceTrainer(
        vocab_size = vocab_size,
        special_tokens = special_tokens,
        initial_alphabet = alphabet,
        limit_alphabet = len(alphabet)
    )

    # Post-processing.
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    # Train tokenizer.
    tokenizer.train(text, trainer)

    # Return tokenizer.
    return tokenizer


def calculate_stats(tokenizer, text):
    """
    Calculates statistics for a given tokenizer and text to derive the optimal vocabulary size.
    For small to medium datasets optimal vocabulary size is determined by value F,
    which should be ~100. For large datasets this law breaks down.

    Parameters:
        tokenizer (Tokenizer): The trained tokenizer.
        text (list): List of text sequences.

    Returns:
        D (float): The divergence of the unigram distribution from uniformity.
        F (int): The minimum frequency of the 95th percentile of the unigram distribution.
        u (float): The average length of the sequences in the text.
    """

    # Initialize variables.
    lengths = list()
    counter = Counter()

    # Loop over text.
    for sentence in tqdm(text, total=len(text), ncols=100):

        # Encode sentence.
        tokens = tokenizer.encode(sentence).tokens

        # Update variables.
        lengths.append(len(tokens))
        counter.update(tokens)

    # Calculate divergence D.
    D = 0.5 * sum(np.array(list(counter.values())) /
                  sum(counter.values()) - 1 / tokenizer.get_vocab_size())

    # Calculate F.
    percentile_95 = np.percentile(list(counter.values()), 95)
    classes_above_percentile = [
        c for c, freq in counter.items() if freq >= percentile_95]
    F = min([counter[c] for c in classes_above_percentile])

    # Calculate u.
    u = np.mean(lengths)

    # Return statistics.
    return D, F, u


def save_stats(D, F, u, vocab_size, stats_file):
    """
    Saves statistics to a JSON file.
    
    Parameters:
        D (float): The divergence of the unigram distribution from uniformity.
        F (int): The minimum frequency of the 95th percentile of the unigram distribution.
        u (float): The average length of the sequences in the text.
        vocab_size (int): The vocabulary size.
        stats_file (str): Path to save stats.
    """

    # Initialize results with statistics.
    results = {
        vocab_size: {
            "D": D,
            "F": F,
            "u": u
        }
    }

    # Check if file exists if so load it.
    if os.path.exists(stats_file):
        with open(stats_file, "r") as json_file:
            existing_data = json.load(json_file)
            existing_data.update(results)
            results = existing_data

    # Save results.
    with open(stats_file, "w") as json_file:
        json.dump(results, json_file, indent=4)


def parse_args():
    """
    Parses command line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """

    # Initialize parser.
    parser = argparse.ArgumentParser(
        description="Train a WordPiece tokenizer on a list of text sequences."
    )

    # Add arguments.
    parser.add_argument("--files", type=str, nargs="+", help="Path to French and English files.", required=True)
    parser.add_argument("--vocab-size", type=int, help="Desired vocab size.", required=True)
    parser.add_argument("--output-path", type=str, help="Path to save tokenizer.", required=True)
    parser.add_argument("--alphabet", type=str, help="Additional alphabet.", default=list())
    parser.add_argument("--stats-file", type=str, default=None, help="Path to save stats.")
    parser.add_argument("--pad-token", type=str, default="<|padding|>", help="Padding token.")
    parser.add_argument("--unk-token", type=str, default="<|unknown|>", help="Unknown token.")
    parser.add_argument("--start-token", type=str, default="<|startofseq|>", help="Start of sequence token.")
    parser.add_argument("--end-token", type=str, default="<|endofseq|>", help="End of sequence token.")
    
    # Return parsed arguments.
    return parser.parse_args()


def main():

    # Parse arguments.
    args = parse_args()

    # Train tokenizer.
    tokenizer = train_tokenizer(
        text = args.files,
        vocab_size = args.vocab_size,
        additional_alphabet = args.alphabet,
        pad_token = args.pad_token,
        unk_token = args.unk_token,
        start_token = args.start_token,
        end_token = args.end_token
    )

    # Save tokenizer.
    tokenizer.save(args.output_path)

    # Calculate statistics.
    text = [read_txt(file) for file in args.files]
    D, F, u = calculate_stats(tokenizer, text)

    # Save statistics if path is given.
    if args.stats_file is not None:
        save_stats(
            D = D,
            F = F,
            u = u,
            vocab_size = tokenizer.get_vocab_size(),
            stats_file = args.stats_file
        )

    # Print statistics. 
    # ↓ - lower is better, ↑ - higher is better.
    print("\n### Statistics ###")
    print(f"VS: {tokenizer.get_vocab_size()}\nD↓: {D:.4f}\nF↑: {F}\nµ↓: {u:.4f}")


if __name__ == "__main__":
    main()
