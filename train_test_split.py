import os
source1 = "/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Richtige_Dataset/Karosserie/KR_Aufbau_Seitenteil"
dest11 = "/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Richtige_Dataset_Karosserie/val/KR_Aufbau_Seitenteil"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
    if np.random.rand(1) < 0.3:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)