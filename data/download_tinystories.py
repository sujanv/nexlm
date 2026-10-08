"""
Download and prepare the TinyStories dataset for training NexLM.
TinyStories is a synthetic dataset of small stories with vocabulary of 3-4 year olds,
ideal for training small language models quickly to produce coherent English.
"""

import os
import sys
import argparse
import requests
from tqdm import tqdm

TINYSTORIES_DATASET_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
)


def download_file(url: str, dest_path: str) -> None:
    print(f"Downloading from {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Downloaded successfully: {dest_path}")


def create_synthetic_tinystories(dest_path: str, num_stories: int = 1000) -> None:
    print(f"Generating synthetic TinyStories subset at {dest_path}...")
    characters = ["Lily", "Tim", "Max", "Mia", "Leo", "Emma", "Sam", "Lucy"]
    toys = ["red ball", "wooden train", "soft teddy bear", "shiny kite", "green balloon"]
    locations = ["in the sunny garden", "at the park", "near the big oak tree", "by the calm river"]
    adjectives = ["happy", "cheerful", "curious", "brave", "kind", "playful"]

    stories = []
    import random
    random.seed(42)

    for i in range(num_stories):
        c1 = random.choice(characters)
        c2 = random.choice([c for c in characters if c != c1])
        toy = random.choice(toys)
        loc = random.choice(locations)
        adj = random.choice(adjectives)

        story = (
            f"Once upon a time, {c1} was playing {loc}. "
            f"{c1} was very {adj} today. "
            f"Suddenly, {c1} saw a {toy} on the grass. "
            f"'{c1}, look what I found!' shouted {c2}, running over with a wide smile. "
            f"They decided to share the {toy} and play together all afternoon. "
            f"When it began to get dark, they waved goodbye and went home for dinner. "
            f"They were best friends forever.\n<|endoftext|>\n"
        )
        stories.append(story)

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("".join(stories))
    print(f"Created {num_stories} synthetic stories ({os.path.getsize(dest_path):,} bytes).")


def main():
    parser = argparse.ArgumentParser(description="Download or generate TinyStories dataset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic tiny dataset offline without network download")
    parser.add_argument("--num_stories", type=int, default=2000, help="Number of synthetic stories to generate if --synthetic")
    parser.add_argument("--output", type=str, default="data/tinystories.txt", help="Output file path")
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_tinystories(args.output, num_stories=args.num_stories)
    else:
        try:
            download_file(TINYSTORIES_DATASET_URL, args.output)
        except Exception as e:
            print(f"Download failed ({e}). Falling back to synthetic story generation...")
            create_synthetic_tinystories(args.output, num_stories=args.num_stories)


if __name__ == "__main__":
    main()
