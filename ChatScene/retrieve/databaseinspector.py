import pickle, json

with open("database_v1.pkl","rb") as f:
    obj = pickle.load(f)

with open("database_v1.json","w") as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)
