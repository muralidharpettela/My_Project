
import gviz_api
# page_template = """
# <html>
#   <head>
#   <title>Static example</title>
#     <script src="http://www.google.com/jsapi" type="text/javascript"></script>
#     <script>
#       google.load("visualization", "1", {packages:["table"]});

#       google.setOnLoadCallback(drawTable);
#       function drawTable() {
#         %(jscode)s
#         var jscode_table = new google.visualization.Table(document.getElementById('table_div_jscode'));
#         jscode_table.draw(jscode_data, {showRowNumber: true});

#         var json_table = new google.visualization.Table(document.getElementById('table_div_json'));
#         var json_data = new google.visualization.DataTable(%(json)s, 0.5);
#         json_table.draw(json_data, {showRowNumber: true});
#       }
#     </script>
#   </head>
#   <body>
#     <H1>Table created using ToJSCode</H1>
#     <div id="table_div_jscode"></div>
#     <H1>Table created using ToJSon</H1>
#     <div id="table_div_json"></div>
#   </body>
# </html>
# """
page_template = """
<html>
  <head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {
        %(jscode)s
        var jscode_table = new google.visualization.PieChart(document.getElementById('piechart'));
        jscode_table.draw(jscode_data);
      }
    </script>
  </head>
  <body>
    <div id="piechart" style="width: 900px; height: 500px;"></div>
  </body>
</html>
"""

def main():
    import math
    import gviz_api
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

    path= "C:\\Users\\pettm\\Downloads\\Audi_dataset"

    for root, dirs, files in os.walk(path, topdown=False):
       for index, name in enumerate(files):
           if name.endswith((".json")):
               #full_path = os.path.join(root, name)
                with open(os.path.join(root, name)) as json_file:
                    try:
                        json_text = json.load(json_file)
                        cat1 = json_text['Teilez.E1']
                        if cat1=='Elektrik':
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
    #print(jsondata.apply(pd.Series.value_counts))
    exp=jsondata.apply(pd.Series.value_counts)
    print(exp.dtypes)
# exp.to_csv('out.csv', sep=',')
# a=jsondata['Category_1'].count()
# print("The total number of images or json files: ",a)

# print(jsondata['Category_1'].value_counts())
# print(jsondata['Category_2'].value_counts())
# print(jsondata['Category_3'].value_counts())
# print(jsondata['Category_4'].value_counts())
# print(jsondata.describe())

#plt.figure(figsize=(16,8))
# figure, axes = plt.subplots(nrows=2, ncols=2)
# a=jsondata['Category_1'].value_counts()
# b=jsondata['Category_2'].value_counts()
# c=jsondata['Category_3'].value_counts()
# d=jsondata['Category_4'].value_counts()

# a.plot.pie(ax=axes[0,0],figsize=(6,6),subplots=True,)
# b.plot.pie(ax=axes[0,1],figsize=(6,6),subplots=True)
# c.plot.pie(ax=axes[1,0],figsize=(6,6),subplots=True)
# d.plot.pie(ax=axes[1,1],figsize=(6,6),subplots=True)


# plt.show()
    description = {"names": ("string", "names"),
                "category_1": ("number", "category_1")}
    #description=[('category_1', 'string'), ('category_2', 'string'), ('category_3', 'string'), ('category_4', 'string')]           
    data=[]
    for row in exp.iterrows():
        row1=row[1].values
        cleanedList = [x for x in row1 if (math.isnan(x) != True)]
        none=len(cleanedList)
        if none==1:
            data.append({"names": row[0],"category_1":cleanedList[0]})  
        
             
    data_table = gviz_api.DataTable(description)
    data_table.LoadData(data)
    # Creating a JavaScript code string
    jscode = data_table.ToJSCode("jscode_data",
                               columns_order=("names", "category_1"),
                               order_by="category_1")
    #jscode = data_table.ToJSCode("jscode_data",columns_order=("names", "category_1"))
    # Creating a JSon string
    #json = data_table.ToJSon(columns_order=("names", "category_1"),
                           #order_by="category_1")
  

    # Putting the JS code and JSon string into the template
    with open("Output1.html", "w") as html:
        print(page_template % vars(),file=html)
    #print(page_template % vars())
    

#print(len(cate1))
#print(len(cate2))
#print(len(cate3))
#print(len(cate4))
#print(jsondata)

if __name__ == "__main__":
  main()


