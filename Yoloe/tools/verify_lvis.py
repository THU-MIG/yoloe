
file = '../datasets/lvis/minival.txt'

with open(file, 'r') as f:
    assert(len(f.readlines()) == 4809)

