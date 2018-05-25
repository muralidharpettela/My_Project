import os
import os, json
import pandas as pd
import matplotlib.pyplot as plt
#jsons_data = pd.DataFrame(columns=['cat1', 'cat2', 'cat3','cat4'])
directory_list = list()
cate1=list()
cate2=list()
cate3=list()
cate4=list()

path= "C:\\Users\\pettm\\Downloads\\Audi_dataset\\24_may_2018\\AU736_PVS_GesFzg_20140828142521\\Elektrik"

for root, dirs, files in os.walk(path, topdown=False):
       for index, name in enumerate(files):
           if name.endswith((".json")):
               #full_path = os.path.join(root, name)
                with open(os.path.join(root, name)) as json_file:
                    try:
                        json_text = json.load(json_file)
                        cat1 = json_text['Teilez.E1']
                        cat2 = json_text['Teilez.E2']
                        cat3 = json_text['Teilez.E3']
                        cat4 = json_text['Teilez.E4']
                        cate1.append(cat1)
                        cate2.append(cat2)
                        cate3.append(cat3)
                        cate4.append(cat4)
                    except:
                        print("That is a corrupted file "+name)

                    #jsons_data.loc[index] = [cat1, cat2, cat3,cat4]

# now that we have the pertinent json data in our DataFrame let's look at it
jsondata=pd.DataFrame(list(zip(cate1, cate2, cate3,cate4)),columns=['Category_1','Category_2', 'Category_3','Category_4'])
#jsondata.to_csv('out.csv')
print(jsondata.apply(pd.Series.value_counts))
exp=jsondata.apply(pd.Series.value_counts)
exp.to_csv('out.csv', sep=',')
a=jsondata['Category_1'].count()
print("The total number of images or json files: ",a)

# print(jsondata['Category_1'].value_counts())
# print(jsondata['Category_2'].value_counts())
# print(jsondata['Category_3'].value_counts())
# print(jsondata['Category_4'].value_counts())
# print(jsondata.describe())

#plt.figure(figsize=(16,8))
figure, axes = plt.subplots(nrows=2, ncols=2)
a=jsondata['Category_1'].value_counts()
b=jsondata['Category_2'].value_counts()
c=jsondata['Category_3'].value_counts()
d=jsondata['Category_4'].value_counts()

a.plot.pie(ax=axes[0,0],figsize=(6,6),subplots=True,)
b.plot.pie(ax=axes[0,1],figsize=(6,6),subplots=True)
c.plot.pie(ax=axes[1,0],figsize=(6,6),subplots=True)
d.plot.pie(ax=axes[1,1],figsize=(6,6),subplots=True)


plt.show()

#print(len(cate1))
#print(len(cate2))
#print(len(cate3))
#print(len(cate4))
#print(jsondata)
