import matplotlib.pyplot as plt
import json

json_path = 'experiments\Baseline_AdamW\eval.json'

with open(json_path, 'r') as json_file:
    data = json.load(json_file)

loss_valid = [0]
for item in data['epoch']:
    loss_valid.append(data['epoch'][item]['loss'])

print(loss_valid)

plt.plot(loss_valid)
plt.ylabel('loss')
plt.show()