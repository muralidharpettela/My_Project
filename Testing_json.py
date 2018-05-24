import os
import os, json
import pandas as pd
jsons_data = pd.DataFrame(columns=['cat1', 'cat2', 'cat3','cat4'])
directory_list = list()
for root, dirs, files in os.walk("C:\\AU335_PVS_GesFzg_20140718142257\\Elektrik", topdown=False):
       #json_files = [pos_json for pos_json in os.listdir(os.path.join(root,name)) if pos_json.endswith('.json')]
       for i in range(5000):
           for index, js in enumerate(json_files):
            with open(os.path.join(path_to_json, js)) as json_file:
                json_text = json.load(json_file)

        # here you need to know the layout of your json and each json has to have
        # the same structure (obviously not the structure I have here)
                cat1 = json_text['Teilez.E1']
                cat2 = json_text['Teilez.E2']
                cat3 = json_text['Teilez.E3']
                cat4= json_text['Teilez.E4']
        # here I push a list of data into a pandas DataFrame at row given by 'index'
       
        jsons_data.loc[i] = [cat1, cat2, cat3,cat4]

# now that we have the pertinent json data in our DataFrame let's look at it
print(jsons_data)


           
           





#path_to_json = 'json/'
