import numpy as np
from project_brain_decoder.io.dataset import make_windows


def test_output_shapes():
    T, C, window_size, stride = 100, 192, 30, 5
    neural = np.random.randn(T, C)
    targets = np.random.randn(T, 2)

    X, y = make_windows(neural, targets, window_size, stride)

    expected_n = (T - window_size) // stride + 1
    assert X.shape == (expected_n, window_size, C)
    assert y.shape == (expected_n, 2)


def test_target_alignment():
    """Target for window i should be the value at the last timestep of that window."""
    T, C, window_size, stride = 50, 4, 10, 1
    neural = np.random.randn(T, C)
    targets = np.arange(T).reshape(-1, 1).astype(float)  # easy to verify

    X, y = make_windows(neural, targets, window_size, stride)

    # First window covers timesteps [0, 9] -> target should be targets[9]
    assert y[0, 0] == targets[window_size - 1, 0]
    # Second window covers [1, 10] -> target should be targets[10]
    assert y[1, 0] == targets[window_size, 0]


def test_stride_correctness():
    """Stride > 1 should skip windows correctly."""
    T, C, window_size, stride = 100, 4, 10, 5
    neural = np.random.randn(T, C)
    targets = np.arange(T).reshape(-1, 1).astype(float)

    X, y = make_windows(neural, targets, window_size, stride)

    # y[0] = targets[9], y[1] = targets[14], y[2] = targets[19], ...
    assert y[0, 0] == targets[window_size - 1, 0]
    assert y[1, 0] == targets[window_size - 1 + stride, 0]
    assert y[2, 0] == targets[window_size - 1 + 2 * stride, 0]


def test_window_content():
    """Each window's content should match the corresponding slice of neural."""
    T, C, window_size, stride = 50, 4, 10, 1
    neural = np.arange(T * C).reshape(T, C).astype(float)
    targets = np.random.randn(T, 2)

    X, y = make_windows(neural, targets, window_size, stride)

    # First window should be neural[0:10]
    np.testing.assert_array_equal(X[0], neural[0:window_size])
    # Second window should be neural[1:11]
    np.testing.assert_array_equal(X[1], neural[1:1 + window_size])