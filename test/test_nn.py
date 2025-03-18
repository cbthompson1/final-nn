from nn.nn import NeuralNetwork
from nn import preprocess
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error


def test_single_forward():
    """
    Run a single forward pass with a small sample to make sure the calculations
    are working as intended.
    """
    nn_arch = [{'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    nn = NeuralNetwork(
        nn_arch,
        lr=10000,
        seed=0,
        batch_size=32,
        epochs=123,
        loss_function='blah'
    )

    # Choose prime numbers to make it a little eaasier to figure out where the
    # numbers are coming from.
    W_curr = np.array([[0.1, 0.2], [0.3, 0.5], [0.7, 0.11]])
    b_curr = np.array([[0.13], [0.17], [0.19]])
    A_prev = np.array([[1, 2], [1, 2]])

    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, 'relu')

    # Example calc: 0.1 * 1 + 0.2 * 1 + 0.13 = 0.43 (position 0,0)
    # Example calc: 0.1 * 2 + 0.2 * 2 + 0.13 = 0.73 (position 1,0)
    expected_Z_curr = np.array([[0.43, 0.73], [0.97, 1.77], [1.  , 1.81]])
    expected_A_curr = np.array([[0.43, 0.73], [0.97, 1.77], [1.  , 1.81]])

    # Because of the data going through relu, a = f(z) = z in this case.
    np.testing.assert_array_equal(expected_A_curr, expected_Z_curr)

    np.testing.assert_array_almost_equal(Z_curr, expected_Z_curr)
    np.testing.assert_array_almost_equal(A_curr, expected_A_curr)

def test_forward():
    """
    At this point it gets too complicated (for me) to run out the math, so
    just confirm the lengths of everything match up.
    """
    nn_arch = [
        {'input_dim': 2, 'output_dim': 3, 'activation': 'relu'},
        {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    nn = NeuralNetwork(
        nn_arch,
        lr=0.01,
        seed=0,
        batch_size=32,
        epochs=10,
        loss_function='blah'
    )

    X = np.array([[1, 2], [3, 4]])

    output, cache = nn.forward(X)

    assert output.shape == (1, 2)
    assert len(cache) == 4

    # Check cache keys
    assert set(['Z1', 'A0', 'Z2', 'A1']) == set(cache.keys())

    # Basic shape checks of cache values
    assert cache['Z1'].shape == (3,2)
    assert cache['A0'].shape == (2,2)
    assert cache['Z2'].shape == (1,2)
    assert cache['A1'].shape == (3,2)

def test_single_backprop():
    """Test a single backprop on the same nn as the forward step."""
    W_curr = np.array([[0.1, 0.2], [0.3, 0.5], [0.7, 0.11]])
    Z_curr = np.array([[0.43, 0.73], [0.97, 1.77], [1, 1.81]])
    A_prev = np.array([[1, 2], [1, 2]])
    dA_curr = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    nn = NeuralNetwork(
        [{'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}],
        lr=0.01,
        seed=0,
        batch_size=32,
        epochs=10,
        loss_function='blah'
    )

    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, Z_curr, A_prev, dA_curr, 'relu')

    # Is this a copy/paste from the output? Yes. However, the math lines up afaict.
    expected_dA_prev = np.array([[0.08914099, 0.07007283], [0.04548765, 0.04162637]])
    expected_dW_curr = np.array([[0.05582816, 0.05582816], [0.07964209, 0.07964209], [0.12166841, 0.12166841]])
    expected_db_curr = np.array([[0.03388385], [0.05477013], [0.0854107]])

    np.testing.assert_array_almost_equal(dA_prev, expected_dA_prev)
    np.testing.assert_array_almost_equal(dW_curr, expected_dW_curr)
    np.testing.assert_array_almost_equal(db_curr, expected_db_curr)

def test_predict():
    """
    Test that predict returns reasonable outputs, also that it is within
    bounds.
    """
    nn_arch = [
        {'input_dim': 2, 'output_dim': 1, 'activation': 'relu'},
        {'input_dim': 1, 'output_dim': 2, 'activation': 'relu'},
    ]
    nn = NeuralNetwork(
        nn_arch,
        lr=0.01,
        seed=0,
        batch_size=32,
        epochs=10,
        loss_function='blah'
    )

    X = np.array([0, 1])
    y_pred = nn.predict(X)

    # Assert shape is right, and min and max are between 0 and 1.
    assert y_pred.shape == (2, 1)
    assert y_pred[0] < y_pred[1]
    assert min(y_pred) >= 0
    assert max(y_pred) <= 1

def test_binary_cross_entropy():
    """
    Assert the binary cross entropy calculation works the same as sklearns.
    """
    nn_arch = [{'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    nn = NeuralNetwork(
        nn_arch,
        lr=10000,
        seed=0,
        batch_size=32,
        epochs=123,
        loss_function='blah'
    )

    y_true = np.array([0,0,0,1])
    y_pred = np.array([0,0,0.99,0.99])
    in_house_loss = nn._binary_cross_entropy(y_true, y_pred)
    sklearn_loss = log_loss(y_true, y_pred)
    assert round(in_house_loss, 5) == round(sklearn_loss, 5)

def test_binary_cross_entropy_backprop():
    """Runthrough an example of backprop and confirm identity."""

    nn_arch = [{'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    nn = NeuralNetwork(
        nn_arch,
        lr=10000,
        seed=0,
        batch_size=32,
        epochs=123,
        loss_function='blah'
    )
    y = np.array([0,0,0,1])
    y_pred = np.array([0,0,0.99,0.99])
    dA = nn._binary_cross_entropy_backprop(y,y_pred)
    expected_dA = np.array([0.25, 0.25, 25, -0.25252525])
    np.testing.assert_array_almost_equal(dA, expected_dA)

def test_mean_squared_error():
    """
    Assert the mean squared error calculation works the same as sklearns.
    """
    nn_arch = [{'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    nn = NeuralNetwork(
        nn_arch,
        lr=10000,
        seed=0,
        batch_size=32,
        epochs=123,
        loss_function='blah'
    )

    y_true = np.array([0,0,0,1])
    y_pred = np.array([0,0,0.99,0.99])
    in_house_loss = nn._mean_squared_error(y_true, y_pred)
    sklearn_loss = mean_squared_error(y_true, y_pred)
    assert round(in_house_loss, 5) == round(sklearn_loss, 5)

def test_mean_squared_error_backprop():
    """Runthrough an example of mse backprop and confirm identity."""

    nn_arch = [{'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    nn = NeuralNetwork(
        nn_arch,
        lr=10000,
        seed=0,
        batch_size=32,
        epochs=123,
        loss_function='blah'
    )
    y = np.array([0,0,0,1])
    y_pred = np.array([0,0,0.99,0.99])
    dA = nn._mean_squared_error_backprop(y,y_pred)
    expected_dA = np.array([0, 0, 0.495, -0.005])
    np.testing.assert_array_almost_equal(dA, expected_dA)

def test_sample_seqs():
    """Test sample seqs upsamples to the correct number."""

    # Create sets of positive and negative class.
    positives = ["yes", "yeah", "yep", "yeehaw", "yuppers"]
    negatives = ["no", "nope"]
    seqs = positives + negatives + positives
    labels = [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True
    ]

    # Resample.
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    positives_count = 0
    negatives_count = 0
    # Confirm within each newly created seq/label they are appropriate.
    for seq, label in zip(sampled_seqs, sampled_labels):
        if seq in positives and label == True:
            positives_count += 1
        elif seq in negatives and label == False:
            negatives_count += 1
        else:
            # Non-sequence values shouldn't occur, or labels don't match.
            raise AssertionError(f"Found mismatched sequence/label.")
    # Count should be the same.
    assert positives_count == negatives_count

def test_one_hot_encode_seqs():
    """
    One hot encoding translates all the base pairs and can handle
    repeats in a sequence.
    """
    test_seq = "ATTCG"
    encoded_seq = preprocess.one_hot_encode_seqs(test_seq)
    expected = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    assert encoded_seq == expected