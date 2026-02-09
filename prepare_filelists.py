"""
Prepare train/val/test filelists from LJSpeech-1.1 metadata.
Split: train=12500, val=100, test=500 (per VITS2 paper Section 3).
Uses random split with fixed seed for reproducibility.
"""
import os
import random

SEED = 1234
DATA_DIR = "data/LJSpeech-1.1"
FILELIST_DIR = "data/filelists"
METADATA = os.path.join(DATA_DIR, "metadata.csv")

TRAIN_SIZE = 12500
VAL_SIZE = 100
TEST_SIZE = 500


def main():
    os.makedirs(FILELIST_DIR, exist_ok=True)

    entries = []
    with open(METADATA, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                file_id = parts[0]
                # Use raw text (2nd column) - normalization handled in code
                # (advices.txt item 2: avoid double normalization)
                text = parts[1]
                wav_path = os.path.join(DATA_DIR, "wavs", f"{file_id}.wav")
                entries.append(f"{wav_path}|{text}")

    assert len(entries) == 13100, f"Expected 13100 entries, got {len(entries)}"

    random.seed(SEED)
    random.shuffle(entries)

    train = entries[:TRAIN_SIZE]
    val = entries[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test = entries[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    assert len(train) == TRAIN_SIZE
    assert len(val) == VAL_SIZE
    assert len(test) == TEST_SIZE

    for name, data in [("ljs_train.txt", train), ("ljs_val.txt", val), ("ljs_test.txt", test)]:
        path = os.path.join(FILELIST_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(entry + "\n")
        print(f"Written {len(data)} entries to {path}")

    print(f"Total: {len(train) + len(val) + len(test)} / {len(entries)}")


if __name__ == "__main__":
    main()
