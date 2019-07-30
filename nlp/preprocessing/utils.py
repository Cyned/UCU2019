import os

from typing import List, Tuple

from config import DATA_DIR


def get_dictionaries() -> Tuple[List[str], List[str]]:
    with open(os.path.join(DATA_DIR, 'word_dictionaries/updated_negative.txt'), 'r') as file:
        negative = [s.strip().lower() for s in file.readlines()]
    with open(os.path.join(DATA_DIR, 'word_dictionaries/updated_positive.txt'), 'r') as file:
        positive = [s.strip().lower() for s in file.readlines()]
    return negative, positive


NEGATIVE, POSITIVE = get_dictionaries()
