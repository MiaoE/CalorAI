import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# counting food types 
def count_food_types(json_file_path):
    food_counter = Counter()

    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)

        # count occurrences of each food type
        for entry in data:
            for food in entry["food type"]:
                food_counter[food] += 1

    return food_counter

# json_file_path = "data.json"
# food_counts = count_food_types(json_file_path)

# for food, count in sorted(food_counts.items(), key=lambda x: -x[1]):
#     print(f"{food}: {count}")

'''
Bread: 98
Grapes: 93
Cherry Tomato: 91
Chicken Breast: 89
Cantaloupe: 83
Strawberries: 78
Blueberries: 67
Sweet Potato: 65
Egg: 62
Broccoli: 57
Apple: 51
Carrot: 50
Honeydew: 44
Clementine: 41
Pineapple: 38
Garlic: 34
Pear: 33
Chives: 30
Cauliflower: 28
Jujube: 26
Orange: 23
Banana: 21
Potato: 20
Raisins: 20
Mushrooms: 15
Onion: 14
'''

# Splitting the dataset 
def split_dataset(json_file_path, train_json, val_json, test_json, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    # read input dataset
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # sort food types alphabetically for consistency
    for entry in data:
        entry["food type"] = sorted(entry["food type"])

    # use the first food type as the stratification label
    for entry in data:
        entry["stratify_label"] = entry["food type"][0]

    # count occurrences of each label
    label_counts = Counter(entry["stratify_label"] for entry in data)

    # identify rare labels (appearing only once)
    rare_labels = {label for label, count in label_counts.items() if count < 2}
    common_data = [entry for entry in data if entry["stratify_label"] not in rare_labels]
    rare_data = [entry for entry in data if entry["stratify_label"] in rare_labels]

    print(f"Identified {len(rare_data)} rare cases that need manual distribution.")

    # stratified sampling on common cases
    train_data, temp_data = train_test_split(
        common_data, stratify=[entry["stratify_label"] for entry in common_data], train_size=train_ratio, random_state=42
    )

    val_data, test_data = train_test_split(
        temp_data, stratify=[entry["stratify_label"] for entry in temp_data], train_size=val_ratio / (val_ratio + test_ratio), random_state=42
    )

    # manually distribute rare cases
    random.shuffle(rare_data)
    for i, entry in enumerate(rare_data):
        if i % 3 == 0:
            train_data.append(entry)
        elif i % 3 == 1:
            val_data.append(entry)
        else:
            test_data.append(entry)

    # Remove stratify labels before saving
    for dataset in [train_data, val_data, test_data]:
        for entry in dataset:
            del entry["stratify_label"]

    # Save splits to JSON files
    with open(train_json, mode='w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)

    with open(val_json, mode='w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)

    with open(test_json, mode='w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)

    print(f"Data split complete. Saved to:\n- {train_json}\n- {val_json}\n- {test_json}")
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    print(f"Manually distributed {len(rare_data)} rare data points.")


def split_dataset2(json_file_path, train_json, val_json, test_json, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Shuffle within each group of 10 data points
    random.seed(42)  # Ensure reproducibility
    train_data, val_data, test_data = [], [], []

    for i in range(0, len(data), 10):
        chunk = data[i:i + 10]  # Take a batch of 10
        if len(chunk) < 10:
            train_data.extend(chunk)  # Put remaining data in train if not a full batch
            continue
        
        random.shuffle(chunk)  # Shuffle the chunk
        
        train_data.extend(chunk[:8])  # First 8 go to training
        val_data.append(chunk[8])  # 9th goes to validation
        test_data.append(chunk[9])  # 10th goes to testing

    # Save splits to JSON files
    with open(train_json, mode='w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)

    with open(val_json, mode='w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)

    with open(test_json, mode='w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Data split complete. Saved to:\n- {train_json}\n- {val_json}\n- {test_json}")
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

split_dataset2("data.json", "train.json", "val.json", "test.json")
