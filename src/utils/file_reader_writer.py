import json
from tqdm import tqdm
from pathlib import Path


def find_files(input_directory: str, extension: str):
    """
    Find all file paths with the given file extension.
    :param input_directory: Find all files under this directory
    :param extension: File extension to match
    :return: file_paths_list: list of file path
    """
    # Get all filenames under the directory
    file_paths_list = []
    for file_path in tqdm(Path(input_directory).glob(f'**/*{extension}')):
        file_paths_list.append(file_path)
    return file_paths_list


def dict2json(output_file_path: str, input_dictionary: dict):
    """
    Write out dictionary as json file
    :param output_file_path: Output json file path
    :param input_dictionary: Input dictionary
    """
    with open(output_file_path, 'w') as fh:
        json.dump(input_dictionary, fh, indent=4)
