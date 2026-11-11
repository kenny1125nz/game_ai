import os
from ocr_config import OUTPUT_DIR
from online_dataset import Generator
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
# Add to sys.path
sys.path.append(parent_dir)

from data_utils import D2RTraininngContext

# 1. Setup Parameters

os.makedirs(OUTPUT_DIR, exist_ok=True)

context = D2RTraininngContext("../data/clean_bg/")
generator = Generator(context)


if __name__ == "__main__":
    count = int(sys.argv[1])
    with open(f"{OUTPUT_DIR}/labels.txt", "w") as f:
        for i in range(count):  # Start with 10k for testing
            # Hybrid logic
            label = generator.get_random_label().upper()

            img = generator.generate_fixed_sample_memory(label)
            img.save(f"{OUTPUT_DIR}/sample_{i}.png")

            f.write(f"sample_{i}.png\t{label}\n")

    print(f"Generated {count} samples in {OUTPUT_DIR}")
