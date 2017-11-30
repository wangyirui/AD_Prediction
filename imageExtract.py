import shutil
import os 

for d in os.listdir('./Image'):
    if d[0] != '.':
        for f in os.listdir('./Image/' + d):
            if f[-3:] == 'nii':
                shutil.move("./Image/" + d + '/' + f, './Image1/' + d)
shutil.rmtree('./Image')

