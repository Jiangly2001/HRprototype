import os
import torch


def get_file_list(directory, extensions=None):
    """
    Given a directory, return a list of its files, sorted alphabetically.

    Parameters
    ----------
    directory : str
        The path to the directory.
    extensions : tuple of str, optional
        Only include files with these extensions.

    Returns
    -------
    list of str
        The paths of the files.

    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extensions:
                if filename.endswith(tuple(extensions)):
                    files.append(os.path.join(root, filename))
            else:
                files.append(os.path.join(root, filename))
    return sorted(files)

def tensor_to_list(tensor):
    return tensor.cpu().detach().numpy().tolist() if isinstance(tensor, torch.Tensor) else tensor


def predict_encoding_size(rates, n_patches, last_encoding_size):
    """
    Predict the size of a tensor after encoding, in bytes.

    Parameters
    ----------
    rates : float
        The compression rate.
    n_patches : int
        The number of patches in the tensor.
    last_encoding_size : int
        The size of the tensor after previous encoding.

    Returns
    -------
    int
        The predicted size of the tensor in bytes.

    """
    return int(last_encoding_size * rates * n_patches)