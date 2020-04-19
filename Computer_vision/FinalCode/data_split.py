from sklearn.model_selection import train_test_split
import json
datas = {"image": [], "label": []}
with open("Label.csv") as reader:
    for i in reader:
        data = i.strip().split(",")
        datas['image'].append(data[1])
        datas['label'].append(data[2])
print(set(datas['label']))
X_train, X_test, y_train, y_test = train_test_split(datas["image"], datas["label"], test_size=0.2, random_state=1)
datas["train"] = (X_train, y_train)
datas['val'] = (X_test, y_test)

with open("data/train.json","w") as train, open("data/test.json",'w') as test:
    for data in [{'path':path, "label":label} for path, label in zip(*datas['train'])]:
        train.write(json.dumps(data)+"\n")
    for data in [{'path': path, "label": label} for (path, label) in zip(*datas['val'])]:
        test.write(json.dumps(data) + "\n")
