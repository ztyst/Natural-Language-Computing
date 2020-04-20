import json
import difflib

data = json.load(open("preproc.json"))
data2 = json.load(open("xxxx"))

for i in range(len(data)):
    if str(data[i]) != str(data2[i]):
        print(data[i])
        print(data2[i])

