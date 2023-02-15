import os
p = "datasets/pet/training/flow/"
for f in os.listdir(p):
    print("Checking in " + p +f)
    amount = 0
    for f2 in os.listdir(p+f):
        if os.path.isfile(p+f + "/"+ f2):
            amount += 1
    if amount < 7:
        print(amount)
