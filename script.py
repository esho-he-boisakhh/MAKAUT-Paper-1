import csv
import os
from sklearn.model_selection import train_test_split

os.makedirs("data/yelp", exist_ok=True)

neg = []
pos = []

with open("train.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[0]
        text = row[1].strip()

        if label == "1":
            neg.append(text)
        else:
            pos.append(text)

# Split train/valid
neg_train, neg_valid = train_test_split(neg, test_size=0.1, random_state=42)
pos_train, pos_valid = train_test_split(pos, test_size=0.1, random_state=42)

# Write CARA format
def write_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

write_file("data/yelp/train1.txt", neg_train)
write_file("data/yelp/train2.txt", pos_train)
write_file("data/yelp/valid1.txt", neg_valid)
write_file("data/yelp/valid2.txt", pos_valid)

print("Done.")
