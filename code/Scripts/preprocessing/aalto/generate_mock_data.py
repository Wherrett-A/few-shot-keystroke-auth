"""Standalone script for generating sample data for testing the aalto pre-processing script"""

import os

import config
import numpy as np
import pandas as pd


def create_mock_data_files():
    """Generates mock data and saves it to a mock data directory"""
    if os.path.exists(config.MOCK_INPUT_DATA_DIR):
        print(f"Mock data directory {config.MOCK_INPUT_DATA_DIR} exists.")
        return

    print(f"Creating Mock data in {config.MOCK_INPUT_DATA_DIR}")
    os.makedirs(config.MOCK_INPUT_DATA_DIR, exist_ok=True)

    words = [
        "hello",
        "world",
        "python",
        "programming",
        "data",
        "science",
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
    ]
    for i in range(config.NUM_MOCK_USERS):
        participant_id = f"{i + 1:03d}"
        all_data_for_user = []

        for j in range(config.SESSIONS_PER_MOCK_USER):
            test_section_id = f"{j + 1:03d}"
            sentence = " ".join(np.random.choice(words, size=np.random.randint(5, 10)))
            base_time = i * 100000 + j * 1000
            events = []
            for k, letter in enumerate(sentence):
                press_time = base_time + k * np.random.randint(100, 200)
                release_time = press_time + np.random.randint(50, 150)
                keystroke_id = f"{k + 1:03d}"
                events.append(
                    {
                        "PARTICIPANT_ID": participant_id,
                        "TEST_SECTION_ID": test_section_id,
                        "SENTENCE": sentence,
                        "USER_INPUT": sentence,
                        "KEYSTROKE_ID": keystroke_id,
                        "PRESS_TIME": press_time,
                        "RELEASE_TIME": release_time,
                        "LETTER": letter,
                        "KEYCODE": ord(letter),
                    }
                )
            all_data_for_user.extend(events)

        df = pd.DataFrame(all_data_for_user)
        file_path = os.path.join(
            config.MOCK_INPUT_DATA_DIR, f"{participant_id}_keystrokes.txt"
        )
        df.to_csv(file_path, sep="\t", index=False)
        print(f"Generated data for user {participant_id}")

    print(f"All data generated successfully for {config.NUM_MOCK_USERS} users")


if __name__ == "__main__":
    create_mock_data_files()
