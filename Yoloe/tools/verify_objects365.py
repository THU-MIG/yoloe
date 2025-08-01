import os
from tqdm import tqdm

dir = '../datasets/Objects365v1/labels/train'

count = 0
corrupt_count = 0
for file in tqdm(os.listdir(dir)):
    with open(f'{dir}/{file}', 'r') as f:
        lines = f.readlines()
    assert(len(lines) == len(set(lines)))
    count += len(lines)
    for line in lines:
        line = line.strip()
        line = line.split()
        line = [float(l) for l in line[1:]]
        assert(all([l <= 1 for l in line]))
        assert(all([l >= 0 for l in line]))

assert(count == 8529995)
        
    