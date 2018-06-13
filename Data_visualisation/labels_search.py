#Appended paths to get the labels of the data and be able to create a datatable using google charts api
#Example labels--See Piecharts folder--Table_data---> labels_path.html
import gviz_api
page_template = """
<html>
  <head>
  <title>Labels Tree</title>
    <script src="http://www.google.com/jsapi" type="text/javascript"></script>
    <script>
      google.load("visualization", "1", {packages:["table"]});

      google.setOnLoadCallback(drawTable);
      function drawTable() {
        %(jscode)s
        var jscode_table = new google.visualization.Table(document.getElementById('table_div_jscode'));
        jscode_table.draw(jscode_data, {showRowNumber: true});

        var json_table = new google.visualization.Table(document.getElementById('table_div_json'));
        var json_data = new google.visualization.DataTable(%(json)s, 0.5);
        json_table.draw(json_data, {showRowNumber: true});
      }
    </script>
  </head>
  <body>
    <H1>Labels paths category wise</H1>
    <div id="table_div_jscode"></div>
    <H1>Labels paths category wise-another method</H1>
    <div id="table_div_json"></div>
  </body>
</html>
"""
def main():
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
    #path= "C:\\Users\\pettm\\Desktop"

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
                            cat=cat1+'-->'+cat2+'-->'+cat3+'-->'+cat4
                            cate1.append(cat)
                            cate2.append(cat2)
                            cate3.append(cat3)
                            cate4.append(cat4)                                               
                        except:
                            print("That is a corrupted file "+name)

                    #jsons_data.loc[index] = [cat1, cat2, cat3,cat4]

    # now that we have the pertinent json data in our DataFrame let's look at it
    jsondata=pd.DataFrame(list(zip(cate1,cate2,cate3,cate4)),columns=['Category_1','Category_2','Category_3','Category_4'])
    #jsondata.to_csv('out.csv')
    #print(jsondata.apply(pd.Series.value_counts))
    exp=jsondata.apply(pd.Series.value_counts)
    with open("Labels.txt", "w") as text_file:
        print(jsondata.apply(pd.Series.value_counts),file=text_file)
    exp.to_csv('out.csv', sep='\t')
    a=jsondata['Category_1'].count()
    print("The total number of images or json files: ",a)

    description = {"names": ("string", "names"),
                "category_1": ("number", "category_1"),
                 }
    #description=[('category_1', 'string'), ('category_2', 'string'), ('category_3', 'string'), ('category_4', 'string')]           
    data=[]
    for row in exp.iterrows():
        row1=row[1].values
        row2=row1[0]
        data.append({"names": row[0],"category_1":row2})
    data_table = gviz_api.DataTable(description)
    data_table.LoadData(data)
    # Creating a JavaScript code string
    jscode = data_table.ToJSCode("jscode_data",
                               columns_order=("names", "category_1"),
                               order_by="category_1")
    # Creating a JSon string
    json = data_table.ToJSon(columns_order=("names", "category_1"),
                           order_by="category_1")

    # Putting the JS code and JSon string into the template
    with open("Output.html", "w") as html:
        print(page_template % vars(),file=html)

if __name__ == "__main__":
  main()






