
import json
import random
from collections import defaultdict
import math

def extract_max_n_per_class_per_day(input_data, n=1):
    grouped_data = defaultdict(lambda: {0: [], 1: []})

    for line in input_data:
        if line.strip():
            obj = json.loads(line)
            timestamp = obj["timestamp"]
            label = obj["label"]
            
            date_only = timestamp[:10] 
            
            extracted_obj = {
                "timestamp": timestamp,
                "label": label,
                "text": obj.get("text", "")
            }
            
            grouped_data[date_only][label].append(extracted_obj)

    selected_data = []
    
    full_n_days = 0
    undersampled_days = 0
    total_class_0 = 0
    total_class_1 = 0
    
    for date, classes in grouped_data.items():
        class_0_items = classes[0]
        class_1_items = classes[1]
        
        if len(class_0_items) >= n and len(class_1_items) >= n:
            full_n_days += 1
        else:
            undersampled_days += 1
            
        if class_0_items:
            sampled_0 = random.sample(class_0_items, min(n, len(class_0_items)))
            total_class_0 += len(sampled_0)
            selected_data.extend(sampled_0)
            
        if class_1_items:
            sampled_1 = random.sample(class_1_items, min(n, len(class_1_items)))
            total_class_1 += len(sampled_1)
            selected_data.extend(sampled_1)

    selected_data.sort(key=lambda x: x["timestamp"])

    print(f"Days with at least {n} articles for both classes: {full_n_days}")
    print(f"Days with undersampling (fewer than {n} articles for at least one class): {undersampled_days}")
    print(f"Total days processed: {len(grouped_data)}")
    print(f"Total Class 0 (Left) articles: {total_class_0}")
    print(f"Total Class 1 (Right) articles: {total_class_1}")
    print(f"Total articles in stream: {len(selected_data)}")

    return selected_data



def undersample_stream(input_data):
    parsed_data = []
    for line in input_data:
        if line.strip(): 
            obj = json.loads(line)
            parsed_data.append({
                "timestamp": obj["timestamp"],
                "label": obj["label"],
                "text": obj["text"]
            })

    class_0 = [item for item in parsed_data if item["label"] == 0]
    class_1 = [item for item in parsed_data if item["label"] == 1]

    print(f"Total Class 0 (Left) articles before balancing: {len(class_0)}")
    print(f"Total Class 1 (Right) articles before balancing: {len(class_1)}")

    min_count = min(len(class_0), len(class_1))
    
    if min_count == 0:
        print("No articles to balance.")
        return []

    balanced_class_0 = random.sample(class_0, min_count)
    balanced_class_1 = random.sample(class_1, min_count)

    balanced_data = balanced_class_0 + balanced_class_1

    balanced_data.sort(key=lambda x: x["timestamp"])

    print(f"Total Class 0 (Left) articles after balancing: {len(balanced_class_0)}")
    print(f"Total Class 1 (Right) articles after balancing: {len(balanced_class_1)}")
    print(f"Total articles in stream: {len(balanced_data)}")

    return balanced_data


def split_text_into_parts(text, parts):
    words = text.split()
    if parts <= 1 or len(words) == 0:
        return [text]
    
    base_len = len(words) // parts
    remainder = len(words) % parts
    chunks = []
    idx = 0
    
    for i in range(parts):
        extra = 1 if i < remainder else 0
        chunk_words = words[idx: idx + base_len + extra]
        idx += base_len + extra
        if not chunk_words:
            chunk_words = words[-1:]
        chunks.append(" ".join(chunk_words))
    
    return chunks



def extract_and_split_minority_per_day(input_data):
    grouped_data = defaultdict(lambda: {0: [], 1: []})

    for line in input_data:
        if line.strip():
            obj = json.loads(line)
            timestamp = obj["timestamp"]
            label = obj["label"]
            date_only = timestamp[:10]
            grouped_data[date_only][label].append({
                "timestamp": timestamp,
                "label": label,
                "text": obj.get("text", "")
            })

    selected_data = []
    skipped_days = 0
    total_days = 0
    total_majority = 0
    total_minority_parts = 0

    for date, classes in grouped_data.items():
        total_days += 1
        count_0 = len(classes[0])
        count_1 = len(classes[1])
        
        if count_0 == 0 or count_1 == 0:
            skipped_days += 1
            continue
        
        majority_label = 0 if count_0 >= count_1 else 1
        minority_label = 1 - majority_label
        
        majority_items = classes[majority_label]
        minority_items = classes[minority_label]
        target_parts = len(majority_items)
        remaining = target_parts
        remaining_articles = len(minority_items)
        
        for item in majority_items:
            selected_data.append(item)
        total_majority += len(majority_items)
        
        for item in minority_items:
            if remaining <= 0:
                break
            parts = math.ceil(remaining / remaining_articles)
            parts = max(1, parts)
            chunks = split_text_into_parts(item["text"], parts)
            chunks = chunks[:remaining]
            for idx, chunk in enumerate(chunks):
                selected_data.append({
                    "timestamp": item["timestamp"],
                    "label": item["label"],
                    "text": chunk
                })
                total_minority_parts += 1
            remaining -= len(chunks)
            remaining_articles -= 1
        
        if remaining > 0 and minority_items:
            last_item = minority_items[-1]
            for _ in range(remaining):
                selected_data.append({
                    "timestamp": last_item["timestamp"],
                    "label": last_item["label"],
                    "text": last_item["text"]
                })
                total_minority_parts += 1
            remaining = 0

    selected_data.sort(key=lambda x: x["timestamp"])

    print(f"Total days processed: {total_days}")
    print(f"Days skipped (missing one class): {skipped_days}")
    print(f"Majority class articles used: {total_majority}")
    print(f"Minority class parts created: {total_minority_parts}")
    print(f"Total articles in stream: {len(selected_data)}")

    return selected_data