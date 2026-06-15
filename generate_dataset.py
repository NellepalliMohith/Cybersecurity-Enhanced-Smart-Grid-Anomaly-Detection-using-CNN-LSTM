import pandas as pd
import numpy as np

np.random.seed(42)
rows = 1000

voltage = []
current = []
power = []
frequency = []
labels = []

for i in range(rows):
    label = np.random.choice([0,1,2], p=[0.6,0.2,0.2])

    if label == 0:  # Normal
        v = np.random.normal(230, 5)
        c = np.random.normal(5, 1)
        f = np.random.normal(50, 0.2)

    elif label == 1:  # Energy Theft (low power manipulation)
        v = np.random.normal(230, 5)
        c = np.random.normal(2, 0.5)  # suspicious low current
        f = np.random.normal(50, 0.2)

    else:  # False Data Injection
        v = np.random.normal(260, 8)  # abnormal voltage
        c = np.random.normal(8, 2)
        f = np.random.normal(48, 1)  # abnormal frequency

    p = v * c

    voltage.append(v)
    current.append(c)
    power.append(p)
    frequency.append(f)
    labels.append(label)

data = pd.DataFrame({
    "voltage": voltage,
    "current": current,
    "power": power,
    "frequency": frequency,
    "label": labels
})

data.to_csv("data/smartgrid_dataset.csv", index=False)

print("Improved dataset generated successfully!")
