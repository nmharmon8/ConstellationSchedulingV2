import json

with open('p.json', 'r') as file:
    data = json.load(file)

print(len(data))

counts = {}
for plane in data:
    if plane['aa_str'] not in counts:
        counts[plane['aa_str']] = 0
    counts[plane['aa_str']] += 1

print(counts)