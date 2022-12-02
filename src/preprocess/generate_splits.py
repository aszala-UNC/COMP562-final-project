import json
import csv


all_data = []

with open("./data/captions.csv", 'r') as f:
    rows = csv.DictReader(f)

    for row in rows:
        all_data.append({ "id": row["image"].replace(".jpg", ""),
                          "image": row["image"],
                          "caption": row["caption"] })

total_size = len(all_data)

train_size = 30_000
val_size = 5_200
test_size = total_size - train_size - val_size

train_dataset = []
val_dataset = []
test_dataset = []

for data in all_data[0:train_size]:
    train_dataset.append(data)

for data in all_data[train_size:train_size+val_size]:
    val_dataset.append(data)

for data in all_data[train_size+val_size:]:
    test_dataset.append(data)


with open("./data/train.json", 'w') as f:
    json.dump(train_dataset, f)

with open("./data/val.json", 'w') as f:
    json.dump(val_dataset, f)
    
with open("./data/test.json", 'w') as f:
    json.dump(test_dataset, f)


print("Total:", total_size)
print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))