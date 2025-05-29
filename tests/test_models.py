"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

''''# @ symbol: decorator; wrap around the function;
    Current decorator is taking a string and tuple. it will use the string to the corresponding part of the tuple.
    e.g. the first time this is tested will assign [[0, 0], [0, 0], [0, 0]] to test the decorator and [0, 0] to the
    expected decorator.'''
@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4])
    ]
)

def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

def test_daily_max_integer():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2, 7],
                          [1, 3, 4],
                          [5, 6, 9]])

    test_result = np.array([5, 6, 9])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_min_integer():
    """Test that mean function works for an array of positive and negative integers."""

    test_input = np.array([[1, -5, 2],
                           [3, 4, -3],
                           [1, 5, 6]])
    test_result = np.array([1, -5, -3])

    npt.assert_array_equal(daily_min(test_input), test_result)

def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
