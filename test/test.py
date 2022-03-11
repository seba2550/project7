# BMI 203 Project 7: Neural Network
# Import necessary dependencies here
import numpy as np
from nn import nn, preprocess


def test_forward():
    pass


def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode(): # Test that the one hot encoding function actually encodes nucleotides the way it's supposed to
    # Straight from the docstring for the one_hot_encode_seqs function
    seq = 'AGA'
    true_encoding = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])

    test_encoding = preprocess.one_hot_encode_seqs(seq)
    assert np.array_equal(true_encoding, test_encoding)
    
    # Check that the encoding is 4 times longer than the original sequence
    assert len(test_encoding) == len(seq) * 4 # In this case should be 12


def test_sample_seqs(): # Test that we're upsampling the least prominent class 
    seqs = ["AGA", "ATC", "TCG", "GCG", "TAA"]
    labels = [True, False, False, False]

    true_seqs = ["AGA", "AGA", "AGA", "ATC", "TCG", "GCG", "TAA"] # This represents a balanced dataset where the least common label and its corresponding sequence have been upsampled
    true_labels = [True, True, True, False, False, False]

    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)

    assert sampled_seqs == true_seqs
    assert sampled_labels == true_labels