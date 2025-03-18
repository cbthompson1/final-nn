# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    
    ##################### EXPLANATION FOR SAMPLING CHOICE #####################
    # I chose to use upsampling with replacement because it does not remove   #
    # any additional information from the samples, as opposed to downsampling #
    ###########################################################################

    # Base case: already balanced
    if sum(labels) == len(labels)//2:
        return seqs, labels

    # Filter into positives and negatives.
    positives = list(filter(lambda obs: obs[0] == True, zip(labels, seqs)))
    positives = list(map(lambda obs: obs[1], positives))
    negatives = list(filter(lambda obs: obs[0] == False, zip(labels, seqs)))
    negatives = list(map(lambda obs: obs[1], negatives))
    
    # Refactor positives/negatives into majority/minority naming convention.
    if len(positives) < len(negatives):
        minority_class = positives
        majority_class = negatives
        minority_label = True
        majority_label = False
    else:
        minority_class = negatives
        majority_class = positives
        minority_label = False
        majority_label = True

    # Resample minority class with replacement until lengths are equal. Combine
    # afterwards into the sequence list to return.
    num_to_sample = len(majority_class) - len(minority_class)
    new_samples = [random.choice(minority_class) for _ in range(num_to_sample)]
    minority_class = np.concatenate((minority_class, new_samples))
    new_seqs = np.concatenate((majority_class, minority_class))

    # Create labels for the minority/majority and concatenate together.
    minority_labels = np.full(fill_value=minority_label, shape=len(minority_class))
    majority_labels = np.full(fill_value=majority_label, shape=len(majority_class))
    new_labels = np.concatenate((majority_labels, minority_labels))

    return new_seqs, new_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    conversion_dict = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
    }
    converted_list = []
    for sequence in seq_arr:
        converted_seq = []
        for base_pair in sequence:
            converted_seq += conversion_dict[base_pair]
        converted_list.append(converted_seq)
    return converted_list