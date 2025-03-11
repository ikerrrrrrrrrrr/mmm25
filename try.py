tup = ['a', 'c', 'e']

lis = {i: x for x, i in enumerate(tup)}

lis['fff'] = 222

del lis['c']

print(lis)

print(sorted(tup))

