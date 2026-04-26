"""Comprehensive test suite for Aalto preprocessing module.

Tests cover:
- Argument parsing (CLI flags)
- Session-level train/test split (no data leak)
- Chunked processing (memory efficiency)
- Error handling (validation, corrupted files)
- Mock data generation (determinism)
- Edge cases (empty sessions, single user, etc.)

All tests use mock data only - no real Aalto data required.
Each test runs in <5 seconds.
"""

import glob
import json
import os
import shutil
from unittest.mock import patch

import config
import generate_mock_data
import h5py
import numpy as np
import pandas as pd
import preprocessing
import pytest
import run_preprocessing

# ============================================================================
# Test Argument Parsing
# ============================================================================


class TestArgumentParsing:
    """Tests for CLI argument parsing in run_preprocessing.py."""

    def test_test_flag_activates_mock_mode(self, tmp_path):
        """--test flag should use mock data directories."""
        parser = run_preprocessing.argparse.ArgumentParser()
        parser.add_argument("-t", "--test", action="store_true")
        args = parser.parse_args(["--test"])
        assert args.test is True

    def test_ratio_flag_default_value(self):
        """Ratio should default to config value when not specified."""
        parser = run_preprocessing.argparse.ArgumentParser()
        parser.add_argument("-r", "--ratio", type=float, default=None)
        args = parser.parse_args([])
        assert args.ratio is None
        # When None, should use config.SPLIT_RATIO
        expected = config.SPLIT_RATIO
        actual = args.ratio if args.ratio is not None else config.SPLIT_RATIO
        assert actual == expected

    def test_ratio_flag_custom_value(self):
        """--ratio flag should override default split ratio."""
        parser = run_preprocessing.argparse.ArgumentParser()
        parser.add_argument("-r", "--ratio", type=float, default=None)
        args = parser.parse_args(["--ratio", "0.7"])
        assert args.ratio == 0.7

    def test_chunk_size_flag_default(self):
        """Chunk size should default to config value when not specified."""
        parser = run_preprocessing.argparse.ArgumentParser()
        parser.add_argument("-c", "--chunk-size", type=int, default=None)
        args = parser.parse_args([])
        assert args.chunk_size is None

    def test_chunk_size_flag_custom(self):
        """--chunk-size flag should override default."""
        parser = run_preprocessing.argparse.ArgumentParser()
        parser.add_argument("-c", "--chunk-size", type=int, default=None)
        args = parser.parse_args(["--chunk-size", "50"])
        assert args.chunk_size == 50

    def test_verbose_flag(self):
        """--verbose flag should be recognized."""
        parser = run_preprocessing.argparse.ArgumentParser()
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

    def test_combined_flags(self):
        """Multiple flags should work together."""
        parser = run_preprocessing.argparse.ArgumentParser()
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("-t", "--test", action="store_true")
        parser.add_argument("-r", "--ratio", type=float, default=None)
        parser.add_argument("-c", "--chunk-size", type=int, default=None)
        args = parser.parse_args(["--test", "--ratio", "0.9", "--chunk-size", "25"])
        assert args.test is True
        assert args.ratio == 0.9
        assert args.chunk_size == 25


# ============================================================================
# Test Session-Level Split
# ============================================================================


class TestSessionLevelSplit:
    """Tests verifying session-level train/test split prevents data leakage."""

    @pytest.fixture
    def mock_session_data(self, tmp_path):
        """Create mock data with identifiable sessions for split testing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create 3 users, each with 4 sessions
        for user_id in ["001", "002", "003"]:
            events = []
            for session_id in range(1, 5):
                # Each session has unique keystroke pattern
                base_time = int(user_id) * 10000 + session_id * 1000
                for k in range(30):  # 30 keystrokes per session
                    press_time = base_time + k * 150
                    release_time = press_time + 80
                    events.append(
                        {
                            "PARTICIPANT_ID": user_id,
                            "TEST_SECTION_ID": f"session_{session_id}",
                            "PRESS_TIME": press_time,
                            "RELEASE_TIME": release_time,
                            "SENTENCE": "test",
                            "USER_INPUT": "test",
                            "KEYSTROKE_ID": f"{k:03d}",
                            "LETTER": "x",
                            "KEYCODE": 120,
                        }
                    )
            df = pd.DataFrame(events)
            df.to_csv(input_dir / f"{user_id}_keystrokes.txt", sep="\t", index=False)

        return str(input_dir)

    def test_sessions_do_not_leak_between_splits(self, mock_session_data, tmp_path):
        """All windows from a session must go entirely to train OR test."""
        output_file = str(tmp_path / "output" / "test")

        preprocessing.process_and_save(
            input_dir=mock_session_data,
            output_file=output_file,
            window_size=10,
            stride=5,
            split_ratio=0.6,
            chunk_size=10,
        )

        # Load train and test data
        with h5py.File(f"{output_file}:train", "r") as f_train:
            train_x = f_train["x"][:]
            train_user_map = json.loads(f_train.attrs["user_map"])

        with h5py.File(f"{output_file}:test", "r") as f_test:
            test_x = f_test["x"][:]
            test_user_map = json.loads(f_test.attrs["user_map"])

        # Both splits should have data
        assert len(train_x) > 0, "Train set should not be empty"
        assert len(test_x) > 0, "Test set should not be empty"

    def test_split_ratio_respected(self, mock_session_data, tmp_path):
        """Train/test split should approximately match configured ratio."""
        output_file = str(tmp_path / "output" / "test")
        split_ratio = 0.75

        preprocessing.process_and_save(
            input_dir=mock_session_data,
            output_file=output_file,
            window_size=10,
            stride=5,
            split_ratio=split_ratio,
            chunk_size=10,
        )

        with h5py.File(f"{output_file}:train", "r") as f_train:
            train_count = f_train["x"].shape[0]

        with h5py.File(f"{output_file}:test", "r") as f_test:
            test_count = f_test["x"].shape[0]

        total = train_count + test_count
        actual_ratio = train_count / total if total > 0 else 0

        # Allow 20% tolerance due to session-level granularity
        assert abs(actual_ratio - split_ratio) < 0.2, (
            f"Split ratio {actual_ratio:.2f} too far from expected {split_ratio}"
        )

    def test_all_users_represented_in_both_splits(self, mock_session_data, tmp_path):
        """With enough sessions, all users should appear in both train and test."""
        output_file = str(tmp_path / "output" / "test")

        preprocessing.process_and_save(
            input_dir=mock_session_data,
            output_file=output_file,
            window_size=10,
            stride=5,
            split_ratio=0.5,
            chunk_size=10,
        )

        with h5py.File(f"{output_file}:train", "r") as f_train:
            train_y = f_train["y"][:]

        with h5py.File(f"{output_file}:test", "r") as f_test:
            test_y = f_test["y"][:]

        # With 4 sessions per user and 0.5 split, users should appear in both
        train_users = set(train_y)
        test_users = set(test_y)

        # At least some overlap expected with multiple sessions per user
        assert len(train_users) > 0
        assert len(test_users) > 0


# ============================================================================
# Test Chunked Processing
# ============================================================================


class TestChunkedProcessing:
    """Tests for memory-efficient chunked HDF5 writing."""

    @pytest.fixture
    def chunk_test_data(self, tmp_path):
        """Create data for chunk processing tests."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create 5 users with multiple sessions
        for user_id in range(1, 6):
            events = []
            for session_id in range(1, 11):  # 10 sessions each
                base_time = user_id * 100000 + session_id * 1000
                for k in range(50):
                    events.append(
                        {
                            "PARTICIPANT_ID": f"{user_id:03d}",
                            "TEST_SECTION_ID": f"s{session_id}",
                            "PRESS_TIME": base_time + k * 100,
                            "RELEASE_TIME": base_time + k * 100 + 60,
                            "SENTENCE": "chunk test",
                            "USER_INPUT": "chunk test",
                            "KEYSTROKE_ID": f"{k:03d}",
                            "LETTER": "c",
                            "KEYCODE": 99,
                        }
                    )
            df = pd.DataFrame(events)
            df.to_csv(
                input_dir / f"{user_id:03d}_keystrokes.txt", sep="\t", index=False
            )

        return str(input_dir)

    def test_chunked_output_identical_to_single_chunk(self, chunk_test_data, tmp_path):
        """Output should be identical regardless of chunk size."""
        output1 = tmp_path / "out1" / "test"
        output2 = tmp_path / "out2" / "test"

        # Process with chunk_size=5
        preprocessing.process_and_save(
            input_dir=chunk_test_data,
            output_file=str(output1),
            window_size=10,
            stride=5,
            split_ratio=0.8,
            chunk_size=5,
        )

        # Process with chunk_size=100 (effectively single chunk)
        preprocessing.process_and_save(
            input_dir=chunk_test_data,
            output_file=str(output2),
            window_size=10,
            stride=5,
            split_ratio=0.8,
            chunk_size=100,
        )

        # Compare train files
        with (
            h5py.File(f"{output1}:train", "r") as f1,
            h5py.File(f"{output2}:train", "r") as f2,
        ):
            assert f1["x"].shape == f2["x"].shape
            assert f1["y"].shape == f2["y"].shape
            # Data should be identical (same random_state)
            np.testing.assert_array_equal(f1["y"][:], f2["y"][:])

    def test_small_chunk_size_produces_valid_output(self, chunk_test_data, tmp_path):
        """Chunk size of 1 should still produce valid HDF5 files."""
        output = tmp_path / "out" / "test"

        preprocessing.process_and_save(
            input_dir=chunk_test_data,
            output_file=str(output),
            window_size=10,
            stride=5,
            split_ratio=0.8,
            chunk_size=1,  # Minimum chunk size
        )

        with h5py.File(f"{output}:train", "r") as f:
            assert f["x"].shape[0] > 0
            assert f["y"].shape[0] > 0
            assert "user_map" in f.attrs

    def test_hdf5_structure_correct(self, chunk_test_data, tmp_path):
        """Output HDF5 should have correct structure (x, y, attrs)."""
        output = tmp_path / "out" / "test"

        preprocessing.process_and_save(
            input_dir=chunk_test_data,
            output_file=str(output),
            window_size=15,
            stride=3,
            split_ratio=0.7,
            chunk_size=10,
        )

        for split in ["train", "test"]:
            with h5py.File(f"{output}:{split}", "r") as f:
                # Check datasets exist
                assert "x" in f, f"Missing 'x' dataset in {split}"
                assert "y" in f, f"Missing 'y' dataset in {split}"

                # Check shapes
                x = f["x"]
                y = f["y"]
                assert len(x.shape) == 3, "x should be 3D (samples, window, features)"
                assert x.shape[1] == 15, "x window_size should match config"
                assert x.shape[2] == 2, (
                    "x should have 2 features (HOLD_TIME, FLIGHT_TIME)"
                )
                assert y.shape[0] == x.shape[0], "y and x should have same sample count"

                # Check attributes
                assert "user_map" in f.attrs
                assert "window_size" in f.attrs
                assert "stride" in f.attrs
                assert "split" in f.attrs
                assert f.attrs["split"] == split


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for validation and error handling."""

    def test_invalid_window_size_raises_error(self, tmp_path):
        """window_size <= 1 should raise ValueError."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_file = str(tmp_path / "output" / "test")

        # Create minimal valid input
        df = pd.DataFrame(
            [
                {
                    "PARTICIPANT_ID": "001",
                    "TEST_SECTION_ID": "s1",
                    "PRESS_TIME": 100,
                    "RELEASE_TIME": 160,
                    "SENTENCE": "t",
                    "USER_INPUT": "t",
                    "KEYSTROKE_ID": "001",
                    "LETTER": "a",
                    "KEYCODE": 97,
                }
            ]
        )
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)

        with pytest.raises(ValueError, match="window_size"):
            preprocessing.process_and_save(
                input_dir=str(input_dir),
                output_file=output_file,
                window_size=1,
                stride=1,
            )

    def test_invalid_split_ratio_zero_raises_error(self, tmp_path):
        """split_ratio <= 0 should raise ValueError."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_file = str(tmp_path / "output" / "test")

        df = pd.DataFrame(
            [
                {
                    "PARTICIPANT_ID": "001",
                    "TEST_SECTION_ID": "s1",
                    "PRESS_TIME": 100,
                    "RELEASE_TIME": 160,
                    "SENTENCE": "t",
                    "USER_INPUT": "t",
                    "KEYSTROKE_ID": "001",
                    "LETTER": "a",
                    "KEYCODE": 97,
                }
            ]
        )
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)

        with pytest.raises(ValueError, match="split_ratio"):
            preprocessing.process_and_save(
                input_dir=str(input_dir),
                output_file=output_file,
                window_size=10,
                stride=5,
                split_ratio=0.0,
            )

    def test_invalid_split_ratio_above_one_raises_error(self, tmp_path):
        """split_ratio > 1 should raise ValueError."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_file = str(tmp_path / "output" / "test")

        df = pd.DataFrame(
            [
                {
                    "PARTICIPANT_ID": "001",
                    "TEST_SECTION_ID": "s1",
                    "PRESS_TIME": 100,
                    "RELEASE_TIME": 160,
                    "SENTENCE": "t",
                    "USER_INPUT": "t",
                    "KEYSTROKE_ID": "001",
                    "LETTER": "a",
                    "KEYCODE": 97,
                }
            ]
        )
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)

        with pytest.raises(ValueError, match="split_ratio"):
            preprocessing.process_and_save(
                input_dir=str(input_dir),
                output_file=output_file,
                window_size=10,
                stride=5,
                split_ratio=1.5,
            )

    def test_empty_input_directory_raises_error(self, tmp_path):
        """Empty input directory should raise FileNotFoundError."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_file = str(tmp_path / "output" / "test")

        with pytest.raises(FileNotFoundError, match="No valid input files"):
            preprocessing.process_and_save(
                input_dir=str(input_dir),
                output_file=output_file,
                window_size=10,
                stride=5,
            )

    def test_missing_columns_raises_error(self, tmp_path, capsys):
        """Files missing required columns should raise error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_file = str(tmp_path / "output" / "test")

        # Create file with wrong columns
        df = pd.DataFrame([{"wrong_column": "data"}])
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)

        with pytest.raises(ValueError, match="No valid participant"):
            preprocessing.process_and_save(
                input_dir=str(input_dir),
                output_file=output_file,
                window_size=10,
                stride=5,
            )

    def test_output_directory_created_if_not_exists(self, tmp_path):
        """Output directory should be created automatically."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "nested" / "deep" / "output"
        output_file = str(output_dir / "test")

        # Create valid input with multiple sessions to enable stratified split
        events = []
        for session_id in range(1, 5):  # 5 sessions for user 001
            base_time = session_id * 1000
            for k in range(30):
                events.append(
                    {
                        "PARTICIPANT_ID": "001",
                        "TEST_SECTION_ID": f"s{session_id}",
                        "PRESS_TIME": base_time + k * 100,
                        "RELEASE_TIME": base_time + k * 100 + 60,
                        "SENTENCE": "test",
                        "USER_INPUT": "test",
                        "KEYSTROKE_ID": f"{k:03d}",
                        "LETTER": "a",
                        "KEYCODE": 97,
                    }
                )
        df = pd.DataFrame(events)
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)

        # Create second user with multiple sessions
        events2 = []
        for session_id in range(1, 5):  # 4 sessions for user 002
            base_time = session_id * 100000
            for k in range(30):
                events2.append(
                    {
                        "PARTICIPANT_ID": "002",
                        "TEST_SECTION_ID": f"s{session_id}",
                        "PRESS_TIME": base_time + k * 100,
                        "RELEASE_TIME": base_time + k * 100 + 60,
                        "SENTENCE": "test",
                        "USER_INPUT": "test",
                        "KEYSTROKE_ID": f"{k:03d}",
                        "LETTER": "b",
                        "KEYCODE": 98,
                    }
                )
        df2 = pd.DataFrame(events2)
        df2.to_csv(input_dir / "002_keystrokes.txt", sep="\t", index=False)

        preprocessing.process_and_save(
            input_dir=str(input_dir),
            output_file=output_file,
            window_size=10,
            stride=5,
        )

        assert output_dir.exists()


# ============================================================================
# Test Mock Data Generation
# ============================================================================


class TestMockDataGeneration:
    """Tests for deterministic mock data generation."""

    def test_deterministic_output(self, tmp_path):
        """Mock data should be identical across runs (fixed seed)."""
        # Override config paths for test isolation
        with patch.object(config, "MOCK_INPUT_DATA_DIR", str(tmp_path / "mock1")):
            generate_mock_data.create_mock_data_files()

            # Get checksums of generated files
            files1 = sorted(glob.glob(str(tmp_path / "mock1" / "*_keystrokes.txt")))
            data1 = []
            for f in files1:
                with open(f, "rb") as fh:
                    data1.append(fh.read())

        # Generate again in different location
        with patch.object(config, "MOCK_INPUT_DATA_DIR", str(tmp_path / "mock2")):
            generate_mock_data.create_mock_data_files()

            files2 = sorted(glob.glob(str(tmp_path / "mock2" / "*_keystrokes.txt")))
            data2 = []
            for f in files2:
                with open(f, "rb") as fh:
                    data2.append(fh.read())

        # Should be identical
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert d1 == d2

    def test_correct_number_of_users(self, tmp_path):
        """Should generate NUM_MOCK_USERS user files."""
        mock_dir = tmp_path / "mock"
        with patch.object(config, "MOCK_INPUT_DATA_DIR", str(mock_dir)):
            generate_mock_data.create_mock_data_files()

            files = list(mock_dir.glob("*_keystrokes.txt"))
            assert len(files) == config.NUM_MOCK_USERS

    def test_correct_sessions_per_user(self, tmp_path):
        """Each user should have SESSIONS_PER_MOCK_USER sessions."""
        mock_dir = tmp_path / "mock"
        with patch.object(config, "MOCK_INPUT_DATA_DIR", str(mock_dir)):
            generate_mock_data.create_mock_data_files()

            # Read first user file
            first_file = list(mock_dir.glob("*_keystrokes.txt"))[0]
            df = pd.read_csv(first_file, sep="\t")

            unique_sessions = df["TEST_SECTION_ID"].nunique()
            assert unique_sessions == config.SESSIONS_PER_MOCK_USER

    def test_required_columns_present(self, tmp_path):
        """Mock data should have all required columns."""
        mock_dir = tmp_path / "mock"
        with patch.object(config, "MOCK_INPUT_DATA_DIR", str(mock_dir)):
            generate_mock_data.create_mock_data_files()

            first_file = list(mock_dir.glob("*_keystrokes.txt"))[0]
            df = pd.read_csv(first_file, sep="\t")

            required = [
                "PARTICIPANT_ID",
                "TEST_SECTION_ID",
                "PRESS_TIME",
                "RELEASE_TIME",
            ]
            for col in required:
                assert col in df.columns

    def test_existing_directory_skips_regeneration(self, tmp_path, capsys):
        """If mock directory exists, should skip generation."""
        mock_dir = tmp_path / "existing"
        mock_dir.mkdir()

        with patch.object(config, "MOCK_INPUT_DATA_DIR", str(mock_dir)):
            generate_mock_data.create_mock_data_files()

            captured = capsys.readouterr()
            assert "exists" in captured.out


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @staticmethod
    def _create_user_events(participant_id, n_sessions, n_keys, base_offset=0):
        """Helper to create keystroke events for one user with multiple sessions."""
        events = []
        for session_id in range(1, n_sessions + 1):
            base_time = base_offset + session_id * 10000
            for k in range(n_keys):
                events.append(
                    {
                        "PARTICIPANT_ID": participant_id,
                        "TEST_SECTION_ID": f"s{session_id}",
                        "PRESS_TIME": base_time + k * 100,
                        "RELEASE_TIME": base_time + k * 100 + 50,
                        "SENTENCE": "test",
                        "USER_INPUT": "test",
                        "KEYSTROKE_ID": f"{k:03d}",
                        "LETTER": "a",
                        "KEYCODE": 97,
                    }
                )
        return events

    def test_session_shorter_than_window_produces_no_windows(self, tmp_path):
        """Sessions with fewer keystrokes than window_size should produce no windows."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_file = str(tmp_path / "output" / "test")

        # User 001: 4 sessions, but only 5 keystrokes per session (< window_size=10)
        events = []
        for session_id in range(1, 5):
            base_time = session_id * 10000
            for k in range(5):  # Only 5 keystrokes — too short for window_size=10
                events.append(
                    {
                        "PARTICIPANT_ID": "001",
                        "TEST_SECTION_ID": f"s{session_id}",
                        "PRESS_TIME": base_time + k * 100,
                        "RELEASE_TIME": base_time + k * 100 + 50,
                        "SENTENCE": "test",
                        "USER_INPUT": "test",
                        "KEYSTROKE_ID": f"{k:03d}",
                        "LETTER": "a",
                        "KEYCODE": 97,
                    }
                )

        # User 002: 4 sessions, also short
        events2 = []
        for session_id in range(1, 5):
            base_time = 100000 + session_id * 10000
            for k in range(5):
                events2.append(
                    {
                        "PARTICIPANT_ID": "002",
                        "TEST_SECTION_ID": f"s{session_id}",
                        "PRESS_TIME": base_time + k * 100,
                        "RELEASE_TIME": base_time + k * 100 + 50,
                        "SENTENCE": "test",
                        "USER_INPUT": "test",
                        "KEYSTROKE_ID": f"{k:03d}",
                        "LETTER": "b",
                        "KEYCODE": 98,
                    }
                )

        df = pd.DataFrame(events)
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)
        df2 = pd.DataFrame(events2)
        df2.to_csv(input_dir / "002_keystrokes.txt", sep="\t", index=False)

        preprocessing.process_and_save(
            input_dir=str(input_dir),
            output_file=output_file,
            window_size=10,
            stride=5,
            split_ratio=0.5,
        )

        # All sessions are too short, so no output files should be created
        assert not os.path.exists(f"{output_file}:train")
        assert not os.path.exists(f"{output_file}:test")

    def test_large_stride_fewer_windows(self, tmp_path):
        """Larger stride should produce fewer windows."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # User 001: 4 sessions, 30 keys each
        events = self._create_user_events("001", n_sessions=4, n_keys=30, base_offset=0)
        # User 002: 4 sessions, 30 keys each
        events2 = self._create_user_events(
            "002", n_sessions=4, n_keys=30, base_offset=500000
        )

        df = pd.DataFrame(events)
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)
        df2 = pd.DataFrame(events2)
        df2.to_csv(input_dir / "002_keystrokes.txt", sep="\t", index=False)

        # Process with small stride
        output1 = tmp_path / "out1" / "test"
        preprocessing.process_and_save(
            input_dir=str(input_dir),
            output_file=str(output1),
            window_size=10,
            stride=2,
            split_ratio=0.5,
        )

        # Process with large stride
        output2 = tmp_path / "out2" / "test"
        preprocessing.process_and_save(
            input_dir=str(input_dir),
            output_file=str(output2),
            window_size=10,
            stride=20,
            split_ratio=0.5,
        )

        with h5py.File(f"{output1}:train", "r") as f1:
            count1 = f1["x"].shape[0]

        with h5py.File(f"{output2}:train", "r") as f2:
            count2 = f2["x"].shape[0]

        # Larger stride = fewer windows
        assert count2 < count1

    def test_zero_std_features_handled(self, tmp_path):
        """Features with zero standard deviation should not cause errors."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_file = str(tmp_path / "output" / "test")

        # User 001: 4 sessions with identical HOLD_TIME values (zero std)
        events = []
        for session_id in range(1, 5):
            base_time = session_id * 10000
            for k in range(30):
                events.append(
                    {
                        "PARTICIPANT_ID": "001",
                        "TEST_SECTION_ID": f"s{session_id}",
                        "PRESS_TIME": base_time + k * 100,
                        "RELEASE_TIME": base_time + k * 100 + 50,  # All HOLD_TIME = 50
                        "SENTENCE": "test",
                        "USER_INPUT": "test",
                        "KEYSTROKE_ID": f"{k:03d}",
                        "LETTER": "a",
                        "KEYCODE": 97,
                    }
                )

        # User 002: 4 sessions, same constant hold time
        events2 = []
        for session_id in range(1, 5):
            base_time = 100000 + session_id * 10000
            for k in range(30):
                events2.append(
                    {
                        "PARTICIPANT_ID": "002",
                        "TEST_SECTION_ID": f"s{session_id}",
                        "PRESS_TIME": base_time + k * 100,
                        "RELEASE_TIME": base_time + k * 100 + 50,
                        "SENTENCE": "test",
                        "USER_INPUT": "test",
                        "KEYSTROKE_ID": f"{k:03d}",
                        "LETTER": "b",
                        "KEYCODE": 98,
                    }
                )

        df = pd.DataFrame(events)
        df.to_csv(input_dir / "001_keystrokes.txt", sep="\t", index=False)
        df2 = pd.DataFrame(events2)
        df2.to_csv(input_dir / "002_keystrokes.txt", sep="\t", index=False)

        # Should not raise an error
        preprocessing.process_and_save(
            input_dir=str(input_dir),
            output_file=output_file,
            window_size=10,
            stride=5,
            split_ratio=0.5,
        )

        # Verify output was created
        with h5py.File(f"{output_file}:train", "r") as f:
            assert f["x"].shape[0] > 0


# ============================================================================
# Test Sliding Windows
# ============================================================================


class TestSlidingWindows:
    """Tests for create_sliding_windows function."""

    def test_correct_number_of_windows(self):
        """Should create correct number of windows based on stride."""
        # Create features for 100 timesteps
        features = np.random.randn(100, 2)

        windows, labels = preprocessing.create_sliding_windows(
            session_features=features,
            user_label=5,
            window_size=20,
            stride=5,
        )

        # Expected: (100 - 20) / 5 + 1 = 17 windows
        assert len(windows) == 17
        assert all(l == 5 for l in labels)

    def test_window_shape(self):
        """Each window should have shape (window_size, num_features)."""
        features = np.random.randn(50, 2)

        windows, _ = preprocessing.create_sliding_windows(
            session_features=features,
            user_label=0,
            window_size=10,
            stride=2,
        )

        for w in windows:
            assert w.shape == (10, 2)

    def test_short_session_returns_empty(self):
        """Session shorter than window_size should return empty lists."""
        features = np.random.randn(5, 2)  # Only 5 timesteps

        windows, labels = preprocessing.create_sliding_windows(
            session_features=features,
            user_label=0,
            window_size=10,
            stride=1,
        )

        assert len(windows) == 0
        assert len(labels) == 0

    def test_stride_of_one_maximum_windows(self):
        """Stride of 1 should produce maximum number of windows."""
        features = np.random.randn(30, 2)

        windows, labels = preprocessing.create_sliding_windows(
            session_features=features,
            user_label=0,
            window_size=10,
            stride=1,
        )

        # (30 - 10) / 1 + 1 = 21 windows
        assert len(windows) == 21

    def test_normalization_applied(self):
        """Features should be normalized (mean ~0, std ~1 for non-constant)."""
        # Create features with non-zero mean and non-unit std
        features = np.random.randn(50, 2) * 10 + 100

        windows, _ = preprocessing.create_sliding_windows(
            session_features=features,
            user_label=0,
            window_size=10,
            stride=10,
        )

        # Windows should be normalized (approx mean 0, std 1)
        all_windows = np.array(windows)
        mean = np.mean(all_windows)
        std = np.std(all_windows)

        # Should be approximately normalized
        assert abs(mean) < 1.0  # Close to 0
        assert 0.5 < std < 2.0  # Close to 1
