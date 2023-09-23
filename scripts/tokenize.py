import argparse
import os
from collections import deque

from tqdm import tqdm

from tokenizers import Tokenizer

from .utils import read_txt


def tokenize(
        tokenizer, 
        french, 
        english, 
        min_length, 
        max_length, 
        unknown_ratio,
        compress_ratio, 
        length_tolerance,
        unk_token,
    ):

    """
    Tokenizes and cleans parallel French and English text sequences.
    Cleaning is done by dropping sequences that are too short or too long,
    have too many unknown tokens, have too many tokens compared to number
    of characters in the sequence and have too big difference in length
    between French and English sequences.

    Parameters:
        tokenizer (Tokenizer): The tokenizer used for tokenization.
        french (deque): Queue containing French text sequences.
        english (deque): Queue containing English text sequences.
        min_length (int): Minimum allowed sequence length.
        max_length (int): Maximum allowed sequence length.
        unknown_ratio (float): Maximum allowed ratio of unknown tokens.
        compress_ratio (float): Minimum allowed compression ratio.
        length_tolerance (float): Maximum allowed length difference ratio between sequences.
        unk_token (str): The unknown token.

    Returns:
        french_tokenized (list): List of tokenized French sequences.
        english_tokenized (list): List of tokenized English sequences.
    """

    # Initialize queues for tokenized sequences.
    french_tokenized = deque()
    english_tokenized = deque()

    # Get unknown token id.
    unk_id = tokenizer.token_to_id(unk_token)

    # Statistics printed later.
    length_skipped = 0
    compress_skipped = 0
    unknown_skipped = 0
    tolerance_skipped = 0

    # Loop over sequences.
    for _ in tqdm(range(len(french)), total=len(french), ncols=100, desc="Tokenizing"):

        # Get leftmost lines from queues.
        french_line = french.popleft()
        english_line = english.popleft()

        # Tokenize lines.
        french_tokens = tokenizer.encode(french_line).ids
        english_tokens = tokenizer.encode(english_line).ids

        # Check for length limits.
        # Since input for decoder will be shifted by 1 thus will be 1 token longer
        # for English, we only need to subtract 1 to add special tokens.
        if (
            min_length > len(french_tokens)
            or len(french_tokens) > max_length
            or min_length > len(english_tokens) 
            or len(english_tokens) > max_length - 1
        ):
            length_skipped += 1
            continue

        # Check for unknown ratio.
        if (
            french_tokens.count(unk_id) > len(french_tokens) * unknown_ratio 
            or english_tokens.count(unk_id) > len(english_tokens) * unknown_ratio
        ):
            unknown_skipped += 1
            continue
        
        # Check for compression ratio.
        if (
            len(list(french_line)) * compress_ratio < len(french_tokens) 
            or len(list(english_line)) * compress_ratio < len(english_tokens)
        ):
            compress_skipped += 1
            continue

        # Check for length tolerance.
        if (
            min(len(french_tokens), len(english_tokens)) * length_tolerance < 
            max(len(french_tokens), len(english_tokens))
        ):
            tolerance_skipped += 1
            continue

        # If everything was fine append tokens to lists.
        french_tokenized.append(french_tokens)
        english_tokenized.append(english_tokens)

    # Check if lengths are the same.
    assert len(french_tokenized) == len(english_tokenized), "Lengths must be the same."

    # Check if lengths are within limits.
    for fr, en in zip(french_tokenized, english_tokenized):
        assert len(fr) <= max_length, f"Length: {len(fr)} must be <= {max_length}."
        assert len(en) <= max_length - 1, f"Length: {len(en)} must be <= {max_length - 1}."

    # Print statistics.
    print("\n##### Tokenizing Stats #####")
    print(f"Length skipped: {length_skipped:,}")
    print(f"Unk. skipped:   {unknown_skipped:,}")
    print(f"Compr. skipped: {compress_skipped:,}")
    print(f"Tol. skipped:   {tolerance_skipped:,}")
    print("-" * 30)
    print(f"Total skipped:  {length_skipped + compress_skipped + unknown_skipped + tolerance_skipped:,}")
    print(f"Total kept:     {len(french_tokenized):,}\n")

    # Return tokenized sequences.
    return french_tokenized, english_tokenized


def pack_sequences(french, english, max_length, sos_token, eos_token):
    """
    Packs sequences into batches of size max_length.
    Batching is done by adding sequences to batch until
    max_length is reached. Shorter sequences from beginning
    of the queue are mixed with longer sequences from the end
    to avoid having too many padding tokens.

    Parameters:
        french (list): List of tokenized French sequences.
        english (list): List of tokenized English sequences.
        max_length (int): Maximum allowed sequence length.
        sos_token (int): Start of sequence token id.
        eos_token (int): End of sequence token id.
        
    Returns:
        french_packed (list): List of packed French sequences.
        english_packed (list): List of packed English sequences.
    """

    # Sort sequences by length and convert to deques.
    french, english = zip(*sorted(zip(french, english), key=lambda x: len(x[0])))
    french, english = deque(french), deque(english)

    # Initialize lists/queues for packed sequences.
    french_packed, english_packed = [], []
    french_current, english_current = deque(), deque()

    # Loop over sequences.
    with tqdm(total=len(french), ncols=100, desc="Packing") as pbar:

        # Loop until all sequences are packed.
        while french:

            # If current sequences are empty, add new ones.
            if len(french_current) == 0 and len(english_current) == 0:
                french_current.extend(french.pop())

                # Add SOS token to English.
                english_current.extend([sos_token] + english.pop())

                # Update progress bar.
                pbar.update(1)
            
            # Add sequences to batch until max_length is reached.
            # For English we need to add sentinel tokens so we subtract 1 from max_length.
            # Total length of English sequence will be max_length + 1.
            while (
                french and len(french_current) + len(french[0]) <= max_length
                and len(english_current) + len(english[0]) <= max_length - 1
                ):

                # Add sequences to current batch.
                french_current.extend(french.popleft())
                english_current.extend(english.popleft())

                # Update progress bar.
                pbar.update(1)

            # Append current batch to packed sequences and add sentinel token to English.
            french_packed.append(list(french_current))
            english_packed.append(list(english_current) + [eos_token])

            # Clear current batch.
            french_current.clear()
            english_current.clear()

    # Check if lengths are the same.
    assert len(french_packed) == len(english_packed), "Lengths must be the same."

    # Check if lengths are within limits and if sentinel tokens are in correct places.
    for fr, en in zip(french_packed, english_packed):
        assert len(fr) <= max_length, f"Length: {len(fr)} must be <= {max_length}."
        assert len(en) <= max_length + 1, f"Length: {len(en)} must be <= {max_length + 1}."
        assert en[0] == sos_token, "First token must be SOS token."
        assert en[-1] == eos_token, "Last token must be EOS token."

    # Calculate padding statistics.
    total_tokens, total_padding = 0, 0
    for fr, en in zip(french_packed, english_packed):
        total_padding += (max_length - len(fr)) + (max_length - len(en) + 1)
        total_tokens += len(fr) + len(en)

    # Print statistics.
    print("\n##### Packing Stats #####")
    print(f"Packed seqs:   {len(french_packed):,}")
    print(f"Non pad toks:  {total_tokens:,}")
    print(f"Padding toks:  {total_padding:,}")
    print(f"Total toks:    {total_tokens + total_padding:,}")
    print(f"Padding perc.: {total_padding / (total_tokens + total_padding) * 100:.2f}%")

    # Return packed sequences.
    return french_packed, english_packed


def save_tokens(tokens, path, file_name):
    """
    Saves tokenized sequences to file.

    Parameters:
        tokens (list): List of tokenized sequences.
        path (str): Path to output directory.
        file_name (str): Name of output file.
    """

    # Join path and file name.
    full_path = os.path.join(path, file_name)

    # Save tokens to file as strings separated by space.
    with open(full_path, "a") as f:
        for line in tokens:
            f.write(" ".join([str(x) for x in line]) + "\n")


def parse_args():
    """
    Parses command line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """

    # Initialize parser.
    parser = argparse.ArgumentParser(
        description="Tokenize French and English sequences."
    )

    # Add arguments.
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--files", type=str, nargs=2,
                        help="Path to French and English files.", required=True)
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--out-names", type=str, default="toknenized.fr tokenized.en",
                        help="Names of created French and English files.")
    parser.add_argument("--min-length", type=int, default=5,
                        help="All sentences shorter than this length will be dropped.")
    parser.add_argument("--max-length", type=int, default=104,
                        help="All sentences greater than this length will be dropped.")
    parser.add_argument("--unknown-ratio", type=float, default=0.1,
                        help="If number of unknown tokens in line is greater than num \
                              of tokens times this ratio, drop line.")
    parser.add_argument("--compress-ratio", type=float, default=0.3,
                        help="If length of characters in line multiplied by this ratio \
                              is less than num of tokens, drop line.")
    parser.add_argument("--length-tolerance", type=float, default=1.5,
                        help="If length of shorter sequence multiplied with this ratio \
                              is greater than length of longer sequence, drop line")
    parser.add_argument("--start-token", type=str, default="<|startofseq|>",
                        help="Start of sequence token.")
    parser.add_argument("--end-token", type=str, default="<|endofseq|>",
                        help="End of sequence token.")
    parser.add_argument("--unknown-token", type=str, default="<|unknown|>",
                        help="Unknown token.")
    
    # Return parsed arguments.
    return parser.parse_args()


def main():

    # Parse arguments.
    args = parse_args()

    # Read tokenizer.
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Read French and English files and convert to deques.
    french, english = read_txt(args.files[0]), read_txt(args.files[1])
    french, english = deque(french), deque(english)

    # Sanity check for lengths.
    assert len(french) == len(english), "Lengths must be the same."

    # Tokenize sequences.
    french_tokenized, english_tokenized = tokenize(
        tokenizer = tokenizer,
        french = french,
        english=english,
        min_length = args.min_length,
        max_length = args.max_length,
        unknown_ratio = args.unknown_ratio,
        compress_ratio = args.compress_ratio,
        length_tolerance = args.length_tolerance,
        unk_token = args.unknown_token
    )
    
    # Pack sequences.
    french_packed, english_packed = pack_sequences(
        french = french_tokenized,
        english = english_tokenized,
        max_length = args.max_length,
        sos_token = tokenizer.token_to_id(args.start_token),
        eos_token = tokenizer.token_to_id(args.end_token)
    )

    # Create output directory if it doesn't exist.
    os.makedirs(args.out_dir, exist_ok=True)

    # Get output file names.
    french_file, english_file = args.out_names.split()

    # Save tokenized sequences to files.
    save_tokens(
        tokens = french_packed,
        path = args.out_dir,
        file_name = french_file
    )

    save_tokens(
        tokens = english_packed,
        path = args.out_dir,
        file_name = english_file
    )
    

if __name__ == '__main__':
    main()
