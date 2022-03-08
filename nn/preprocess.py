# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from collections import Counter


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    one_hot_seqs = [] # Initialize empty list for placing the encodings
    for i in seq_arr: # Iterate over the sequence and manually encode the values for each nucleotide
        if i == "A":
            one_hot_seqs.append([1, 0, 0, 0])
        if i == "T":
            one_hot_seqs.append([0, 1, 0, 0])
        if i == "C":
            one_hot_seqs.append([0, 0, 1, 0])
        if i == "G":
            one_hot_seqs.append([0, 0, 0, 1])
    
    one_hot_seqs = np.array([item for sublist in one_hot_seqs for item in sublist]) # This flattens the array into a single list
    return one_hot_seqs # Get the final result


def sample_seqs(
        seqs: List[str],
        labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
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
    np.random.seed(42)
    sampled_seqs = [] # Initialize empty lists for holding the sampled seqs and labels (balanced)
    sampled_labels = []

    # Create instance of a counter and get the label counts + the predominant label
    label_counter = Counter(labels)
    max_label = np.max(label_counter.values())

    sampled_idxs = []
    sampled_idxs = np.array(sampled_idxs) # Initialize empty numpy array where we'll place the new indices (i.e. accounting for class imbalance)

    # Iterate over the labels and their respective counts
    for label, count in label_counter.items():
        idxs = np.where(labels == label)[0]
        if count != max_label: # Executes if we're looking at the label that IS NOT predominant
            idxs = np.random.choice(idxs, max_label, replace = True) # Sample randomly (with replacement) from this label and make it the same size of the predominant label. We've essentially upsampled the minority class to balance the data.
        sampled_idxs = np.concatenate((sampled_idxs, idxs)) # Collect the indices for the sampled labels. On one iteration this'll get the labels for the predominant label, and on the other iteration it'll grab the upsampled labels (given that the conditional block executed)
    
    # Use the indices to actually sample sequences and their respective labels
    sampled_seqs = seqs[sampled_idxs] 
    sampled_labels = labels[sampled_idxs]

    return sampled_seqs, sampled_labels 



