import re
import pickle
import os
from datasets import Dataset, load_from_disk
from pprint import pprint
from datasketch import MinHash


save_path = "scenarios.pkl"

# Load existing data if file exists
if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        generateds = pickle.load(f)

# --- Configuration ---
NUM_PERM = 128
JACCARD_THRESHOLD = 0.7
N_GRAM_SIZE = 2


# --- Helper Functions ---

def normalize_text(text):
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_shingles(text, n=1):
    """Create a set of word n-grams from text."""
    # The original text is already normalized
    words = text.split()
    # This creates a set of single words (a "bag of words")
    return { " ".join(words[i:i+n]) for i in range(len(words) - n + 1) }

# --- Map Function for Deduplication ---
def find_duplicates(example, idx, hashes):
    """
    Finds duplicates for a single example by comparing its MinHash
    with all previous examples' hashes.
    """
    text = normalize_text(example['text'])
    shingles = get_shingles(text)
    
    # Return early for empty text
    if not shingles:
        return {'duplicate_id': -1}
        
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles:
        m.update(s.encode('utf8'))

    # Compare with all previous documents
    for i in range(idx):
        if hashes[i] is not None:
            jaccard = m.jaccard(hashes[i])
            if jaccard > JACCARD_THRESHOLD:
                # Found a duplicate, mark it with the ID of the original
                return {'duplicate_id': i}
    
    # No duplicate found, store its hash for future comparisons
    hashes[idx] = m
    return {'duplicate_id': -1} # -1 indicates it's not a duplicate

# --- Main Execution ---

dataset = load_from_disk('toxsyn_1k_with_prompts')
df = dataset.to_pandas()
df = df[~df['consulta'].isnull()]
data = df['consulta'].values

for i in range(1):
    data = {"text": data}
    ds = Dataset.from_dict(data)

    minhashes = [None] * len(ds)

    # 2. Map the function to find duplicates
    #    The `with_indices=True` argument passes the row index to our function.
    #    `fn_kwargs` passes the shared minhashes list.
    updated_ds = ds.map(
        find_duplicates,
        with_indices=True,
        fn_kwargs={'hashes': minhashes},
        num_proc=1 # Important: Must be single-process to share the `hashes` list
    )

    # 3. Filter out the duplicates
    deduplicated_ds = updated_ds.filter(lambda example: example['duplicate_id'] == -1)
    duplicated_ds = updated_ds.filter(lambda example: example['duplicate_id'] != -1)

    # --- Results ---
    print("--- Dataset with Duplicate Flags ---")
    # for item in updated_ds:
    #     print(f"ID: {item['duplicate_id']}\tText: {item['text']}")

    print("\n--- Deduplicated Dataset ---")
    print(f"Original size: {len(ds)}")
    if len(deduplicated_ds) < 50:
        print('\n'*10)
        for item in duplicated_ds['text']:
            print(item)
    print(f"Deduplicated size: {len(deduplicated_ds)}")

    # updated_ds.save_to_disk(split)
    # exit()




# for split, ds in dataset_sec_dang.items():
#     print('Running deduplication on: ', split)
#     # 1. Create a list to store MinHash objects across map calls
#     minhashes = [None] * len(ds)

#     # 2. Map the function to find duplicates
#     #    The `with_indices=True` argument passes the row index to our function.
#     #    `fn_kwargs` passes the shared minhashes list.
#     updated_ds = ds.map(
#         find_duplicates,
#         with_indices=True,
#         fn_kwargs={'hashes': minhashes},
#         num_proc=1 # Important: Must be single-process to share the `hashes` list
#     )

#     # 3. Filter out the duplicates
#     deduplicated_ds = updated_ds.filter(lambda example: example['duplicate_id'] == -1)

#     # --- Results ---
#     print("--- Dataset with Duplicate Flags ---")
#     # for item in updated_ds:
#     #     print(f"ID: {item['duplicate_id']}\tText: {item['text']}")

#     print("\n--- Deduplicated Dataset ---")
#     print(f"Original size: {len(ds)}")
#     print(f"Deduplicated size: {len(deduplicated_ds)}")
#     # for item in deduplicated_ds['text']:
#     #     print(item)

#     updated_ds.save_to_disk(split)

