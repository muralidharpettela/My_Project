'''
Displays Images+Info from json (only x.json) in specified folder. Creates a new y.json file that stores the data from the
x.json + the coordinates of a click event (=EOS Point). Shortcuts may vary on different systems.
Use print(key) to identifie keycodes on your system
'''


import cv2
import os
import sys
import json

#Key values change when NUM_LOCK is activated. Keep NUM_LOCK deactivated!!!

#LINUX:
P_KEY =112 #1048688
#D_KEY=100
Q_KEY=113
RIGHT=100
LEFT=97
UP=119
DOWN=115
SHIFT=226
SPACE=32
BCKSPACE=8
E=101
M=109
I=105
H=104
#C:\\Users\\pettm\\Downloads\\Audi_dataset\\14_may_2018\\AU334_PVS_GesFzg_20140721072917\\Elektrik\\Testwoche_04,0\\
#C:\Users\pettm\Downloads\Audi_dataset\14_may_2018\AU334_PVS_GesFzg_20140721072917\Elektrik\Testwoche_04,0
img_json_folder_path="C:\\Users\\pettm\\Downloads\\Audi_dataset\\24_may_2018_2\\AU736_PVS_GesFzg_20140828142521\\Elektrik\\Testwoche_12,2\\"



def run(im,json_new_path,json_old_path, img_name, length_basename, index_basename):
    coords=[]
    #im_disp = im.copy()
    im_draw = im.copy()
    #im_draw2=im.copy()
    window_name = "Select objects to be tracked here."
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 1200)
    cv2.putText(im_draw, img_name+"  ("+str(index_basename)+"/"+str(length_basename)+")", (10, 50),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 255, 255),4)
    cv2.putText(im_draw, "Press [space] to save; [a] to go back [s=-20]; [d] to go on [w=+20]; [BCKSPACE] to delete ", (10, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),4)

    try:
        with open(json_old_path) as imgdata_old_to_read:
            imgdata_old_to_read_open = json.load(imgdata_old_to_read)
            # cv2.putText(im_draw, "Sichtweite= " + str(imgdata_old_to_read_open["distance"]) + "m", (10, 600),
            #             cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 20)
            cv2.putText(im_draw, "labels= " + str(imgdata_old_to_read_open["Teilez.E1"]) +','+ str(imgdata_old_to_read_open["Teilez.E2"]) +','+ str(imgdata_old_to_read_open["Teilez.E3"]) +','+ str(imgdata_old_to_read_open["Teilez.E4"]) +',', (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    except:
        print("File " + img_name + ".json seems to be corrupted")
        pass
    if os.path.isfile(json_new_path):
        print ("Already labeled")
        try:
            with open(json_new_path) as imgdata_to_read:
                imgdata_to_read_open = json.load(imgdata_to_read)
                print(imgdata_to_read_open["coordEndOfSightPoint"])
            cv2.circle(im_draw, (imgdata_to_read_open["coordEndOfSightPoint"][0],imgdata_to_read_open["coordEndOfSightPoint"][1]), 10,(0, 255, 0), thickness=10, lineType=8, shift=0)

        except:
            print("File "+img_name+"y.json seems to be corrupted")
    cv2.imshow(window_name, im_draw)

    run.mouse_down = False

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("EVENT_LBUTTONDOWN")
            x_coord=x
            y_coord=y
            coords.append((x_coord,y_coord))
            print(coords)

    cv2.setMouseCallback(window_name, callback)

    while True:
        key = cv2.waitKey(1000000000)

        #Use print(key) to identifie keycodes on your system
        print(key)

        if key == E:
            return(coords, key)
        if key == H:
            return(coords, key)
        if key == M:
            return(coords, key)
        if key == I:
            return(coords, key)
        if key == SPACE:
            return(coords, key)
        if key == RIGHT:
            return (coords, key)
        if key == LEFT:
            return (coords, key)
        if key == UP:
            return (coords, key)
        if key == DOWN:
            return (coords, key)
        if key == BCKSPACE:
            return (coords, key)
        elif key == Q_KEY:
            sys.exit(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    basename = []
    index_basename=0

    for filename in os.listdir(img_json_folder_path):
        if filename.endswith((".json")):
            basename.append(os.path.splitext(filename)[0])
            basename = sorted(basename)

    while True:
        
        image_path=img_json_folder_path+basename[index_basename]+".jpg"
        json_path=img_json_folder_path+basename[index_basename]+".json"
        json_new_path=img_json_folder_path+basename[index_basename]+"y.json"
        name=basename[index_basename]
        try:
            im = cv2.imread(image_path)
            print(image_path)
        except:
            print("Cannot read image --> exiting.")
            exit()
        print(basename[index_basename])
        coords, key = run(im, json_new_path,json_path, basename[index_basename], len(basename), index_basename )
        #space key is used to add the category as easy
        if key==SPACE:
            print ("save json and load new image")
            print("coord" +str(coords))
            try:
                #cv2.imwrite('C:\\Users\\pettm\\Desktop\\Data\\'+str(name)+'.jpg',im)
                with open (json_path) as data_to_read:
                    data_to_read_open=json.load(data_to_read)
                    data_to_read_open['CATEGORY'] = 'EASY'
                    
                with open (json_path, 'w') as data_destination:
                    #data_to_read_open["coordEndOfSightPoint"]=[coords[0][0],coords[0][1]]
                    json.dump(data_to_read_open,data_destination)
                    index_basename+=1
            except:
                print("DONE!!!")
        elif key==E:
            print("Category:easy had labelled")
            try:
                with open (json_path) as data_to_read:
                    data_to_read_open=json.load(data_to_read)
                    data_to_read_open['CATEGORY'] = 'EASY'
                with open (json_new_path, 'w') as data_destination:
                    json.dump(data_to_read_open,data_destination)
                    index_basename+=1
            except:
                print("DONE!!!")
        elif key==M:
            print("Category:Medium had labelled")
            try:
                with open (json_path) as data_to_read:
                    data_to_read_open=json.load(data_to_read)
                    data_to_read_open['CATEGORY'] = 'MEDIUM'
                with open (json_new_path, 'w') as data_destination:
                    json.dump(data_to_read_open,data_destination)
                    index_basename+=1
            except:
                print("DONE!!!")
        elif key==H:
            print("Category:Hard had labelled")
            try:
                with open (json_path) as data_to_read:
                    data_to_read_open=json.load(data_to_read)
                    data_to_read_open['CATEGORY'] = 'HARD'
                with open (json_new_path, 'w') as data_destination:
                    json.dump(data_to_read_open,data_destination)
                    index_basename+=1
            except:
                print("DONE!!!")
        elif key==I:
            print("Category:Impossible had labelled")
            try:
                with open (json_path) as data_to_read:
                    data_to_read_open=json.load(data_to_read)
                    data_to_read_open['CATEGORY'] = 'IMPOSSIBLE'
                with open (json_new_path, 'w') as data_destination:
                    json.dump(data_to_read_open,data_destination)
                    index_basename+=1
            except:
                print("DONE!!!")
        elif key==BCKSPACE:
            try:
                os.remove(json_new_path)
                print ("removed: "+json_new_path)
            except:
                print("COULD NOT remove: "+json_new_path)
        elif key==RIGHT:
            print("Load new Image WITHOUT saving")
            index_basename += 1
        elif key==LEFT:
            print("One Image back WITHOUT saving")
            index_basename-=1
        elif key == UP:
            print("One Image back WITHOUT saving")
            index_basename += 20
        elif key == DOWN:
            print("One Image back WITHOUT saving")
            index_basename -= 20
