"""Pytest fixtures for Aalto preprocessing tests."""

import pytest
import os
import sys

# Ensure the aalto module can be imported by tests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


@pytest.fixture
def mock_input_dir():
    """Return path to mock input data directory."""
    return config.MOCK_INPUT_DATA_DIR


@pytest.fixture
def mock_output_file():
    """Return path to mock output data file."""
    return config.MOCK_OUTPUT_DATA_FILE


@pytest.fixture
def default_config():
    """Return default configuration values used by tests."""
    return {
        "window_size": getattr(config, "WINDOW_SIZE", None),
        "stride": getattr(config, "STRIDE", None),
        "split_ratio": getattr(config, "SPLIT_RATIO", None),
        "num_mock_users": getattr(config, "NUM_MOCK_USERS", None),
        "sessions_per_user": getattr(config, "SESSIONS_PER_MOCK_USER", None),
    }
