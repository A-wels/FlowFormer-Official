import os
import shutil
p = "datasets/pet/training/clean/"
for f in os.listdir(p):
    amount = 0
    for f2 in os.listdir(p+f):
        if os.path.isfile(p+f + "/"+ f2):
            amount += 1
            if f2.endswith(".flo"):
                print("Found .flo file")
                print(p+f + "/"+ f2)
                os.remove(p+f + "/"+ f2)
    if amount < 8:
        print("Checking in " + p +f)
        print(amount)
        shutil.rmtree(p+f)
    else:
        print("Checking in " + p +f)
        print(amount)