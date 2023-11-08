import argparse
import os
import random
import shutil

import torch
from tqdm import tqdm

from .dataset import TranslatorDataset

# Constants to store temp files.
TEMP_DIR = "_temp"
FR_TEMP_SHARD = "_fr_temp_shard.pt"
EN_TEMP_SHARD = "_en_temp_shard.pt"
EN_TEMP_MERGED = "english_tensor.pt"
FR_TEMP_MERGED = "french_tensor.pt"


def texts_to_tensor_shards(path, temp_path, max_length):
    """
    Converts text files to tensor shards. 
    This is done to avoid memory issues (at least on my machine).

    Parameters:
        path (str): Path to text files.
        temp_path (str): Path to store temp files.
        max_length (int): Maximum length of a sequence.

    Returns:
        file_counter (int): Number of files.
    """

    # Count the number of files. Used for merging.
    file_counter = 0

    # Loop over files.
    for root, _, files in os.walk(path):

        # Sort files to ensure that the order is the same for both languages.
        fr_files = sorted([file for file in files if file.endswith(".fr")])
        en_files = sorted([file for file in files if file.endswith(".en")])

        # Sanity length check.
        assert len(fr_files) == len(en_files), "Number of French and English files must be equal."
        print(f"\nFound {len(fr_files) + len(en_files)} files.")

        # Loop over French and English files.
        for fr_file, en_file in zip(fr_files, en_files):

            # Open both files.
            with (
                open(os.path.join(root, fr_file), "r") as fr,
                open(os.path.join(root, en_file), "r") as en
            ):  
                # Read lines.
                fr_lines = fr.readlines()
                en_lines = en.readlines()

                # Sanity length check.
                assert len(fr_lines) == len(en_lines), "Number of French and English lines must be equal."

                # List to store tokens.
                fr_tokens, en_tokens = [], []
                
                # Loop over lines.
                for f, e in tqdm(zip(fr_lines, en_lines), total=len(fr_lines), ncols=100):

                    # Convert each line to a list of integers.
                    fr_line = [int(x) for x in f.strip().split()]
                    en_line = [int(x) for x in e.strip().split()]

                    # Check if the length of the sequence is in desired range.
                    if len(fr_line) > max_length or len(en_line) > max_length + 1:
                        continue

                    # Add padding to the end of the sequences.
                    fr_padding = [0] * (max_length - len(fr_line))
                    en_padding = [0] * (max_length - len(en_line) + 1)

                    # Append to the list and add padding.
                    fr_tokens.append(fr_line + fr_padding)
                    en_tokens.append(en_line + en_padding)

                fr_tokens_tensor = torch.tensor(fr_tokens, dtype=torch.long)
                en_tokens_tensor = torch.tensor(en_tokens, dtype=torch.long)

                # Save temporary tensor shards.
                torch.save(fr_tokens_tensor, os.path.join(temp_path, f"{file_counter}{FR_TEMP_SHARD}"))
                torch.save(en_tokens_tensor, os.path.join(temp_path, f"{file_counter}{EN_TEMP_SHARD}"))
                
                # Increment file counter.
                file_counter += 1

    # Return the number of files.
    return file_counter


def merge_and_save_tensor_shards(path, file_counter):
    """
    Merges tensor shards into a single tensor.
    
    Parameters:
        path (str): Path to tensor shards.
        file_counter (int): Number of files.
    """

    # List to merge tensor shards.
    french_shards, english_shards = [], []

    # Loop over tensor shards.
    for i in range(file_counter):

        # Load tensor shards.
        fr = torch.load(os.path.join(path, f"{i}{FR_TEMP_SHARD}"))
        en = torch.load(os.path.join(path, f"{i}{EN_TEMP_SHARD}"))

        # Append to list the final list.
        french_shards.append(fr)
        english_shards.append(en)

        # Remove temp files.
        os.remove(os.path.join(path, f"{i}{FR_TEMP_SHARD}"))
        os.remove(os.path.join(path, f"{i}{EN_TEMP_SHARD}"))

    # Concatenate the shards.
    french_tensor = torch.cat(french_shards, dim=0)
    english_tensor = torch.cat(english_shards, dim=0)

    # Save the temp merged tensors.
    torch.save(french_tensor, os.path.join(path, FR_TEMP_MERGED))
    torch.save(english_tensor, os.path.join(path, EN_TEMP_MERGED))


def split_and_save_tensor(tensor_path, train_percent, dev_percent, seed):
    """
    Splits a tensor into train, dev, and test sets and saves them.

    Parameters:
        tensor_path (str): Path to tensor.
        train_percent (float): Percentage of data to use for training.
        dev_percent (float): Percentage of data to use for dev.
        seed (int): Random seed.
    """

    # Set seed to ensure reproducibility for another run.
    if seed is not None:
        random.seed(seed)

    # Load tensor.
    tensor = torch.load(tensor_path)

    # Calculate data sizes based on percentages.
    total_size = len(tensor)
    train_size = int(train_percent * total_size)
    dev_size = int(dev_percent * total_size)

    # Generate random indices for shuffling.
    indices = list(range(total_size))
    random.shuffle(indices)

    # Split the indices into train, dev, and test sets.
    train_indices = indices[:train_size]
    dev_indices = indices[train_size:(train_size + dev_size)]
    test_indices = indices[(train_size + dev_size):]

    # Use the shuffled indices to split the tensors.
    train_tensor = tensor[train_indices]
    dev_tensor = tensor[dev_indices]
    test_tensor = tensor[test_indices]

    # Save the tensors.
    torch.save(train_tensor, tensor_path.replace(".pt", "_train.pt"))
    torch.save(dev_tensor, tensor_path.replace(".pt", "_dev.pt"))
    torch.save(test_tensor, tensor_path.replace(".pt", "_test.pt"))

    # Remove the original tensor.
    os.remove(tensor_path)


def make_dataset(french_path, english_path, out_file):
    """
    Creates a TranslatorDataset and saves it.
    
    Parameters:
        french_path (str): Path to French tensor.
        english_path (str): Path to English tensor.
        out_file (str): Path to save the dataset.
    """

    # Load French and English tensors.
    french_tensor = torch.load(french_path)
    english_tensor = torch.load(english_path)

    # Create the dataset.
    dataset = TranslatorDataset(french_tensor, english_tensor)

    # Save final dataset.
    torch.save(dataset, out_file)


def parse_args():
    """
    Parses command line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Create a dataset from French and English text files."
    )

    # Add arguments.
    parser.add_argument("--shards", type=str, required=True,
                        help="Path to French and English files. Files must end with .fr and .en")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Path to save the dataset.")
    parser.add_argument("--out-names", type=str, default="train.pt dev.pt test.pt",
                        help="Names of the output files.")
    parser.add_argument("--max-length", type=int, default=104,
                        help="Maximum length of a sequence.")
    parser.add_argument("--temp-dir", type=str, default=".",
                        help="Path to store temp files.")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.9, 0.05, 0.05],
                        help="Train, dev, test split ratios.")
    parser.add_argument("--seed", type=int, default=2018,
                        help="Random seed.")
    
    # Return parsed arguments.
    return parser.parse_args()


def main():
    # Parse arguments.
    args = parse_args()

    # Create temp directory.
    temp_dir = os.path.join(args.temp_dir, TEMP_DIR)
    os.makedirs(temp_dir, exist_ok=True)

    # Convert text files to tensor shards.
    file_counter = texts_to_tensor_shards(args.shards, temp_dir, args.max_length)

    # Merge tensor shards.
    merge_and_save_tensor_shards(temp_dir, file_counter)

    # Split tensors into train, dev, and test sets.
    french_path = os.path.join(temp_dir, FR_TEMP_MERGED)
    english_path = os.path.join(temp_dir, EN_TEMP_MERGED)

    # Split and save tensors into train, dev, and test.
    split_and_save_tensor(
        tensor_path = french_path, 
        train_percent = args.ratios[0], 
        dev_percent = args.ratios[1], 
        seed = args.seed
    )
    split_and_save_tensor(
        tensor_path = english_path, 
        train_percent = args.ratios[0], 
        dev_percent = args.ratios[1], 
        seed = args.seed
    )

    # Make directories for the final dataset.
    os.makedirs(args.out_dir, exist_ok=True)

    # Create the final datasets.
    for split, name in zip(["_train.pt", "_dev.pt", "_test.pt"], args.out_names.split()):
        out_file = os.path.join(args.out_dir, name)

        # Load tensors and create the dataset.
        make_dataset(
            french_path = french_path.replace(".pt", split),
            english_path = english_path.replace(".pt", split),
            out_file = out_file
        )

        # Remove temp files.
        os.remove(french_path.replace(".pt", split))
        os.remove(english_path.replace(".pt", split))

    # Remove final temp directory.
    shutil.rmtree(temp_dir)
        

if __name__ == "__main__":
    main()