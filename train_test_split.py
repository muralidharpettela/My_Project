import os
source1 = "C:\\Users\\pettm\\Desktop\\Richtige_Dataset\\Fahrwerk_Aggregate\\VA"
dest11 = "C:\\Users\\pettm\\Desktop\\Richtige_Dataset_Fahrwerk\\val\\\VA"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
    if np.random.rand(1) < 0.3:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)