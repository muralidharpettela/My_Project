import cv2
import os
import sys
import json


img_json_folder_path="C:\\Users\\pettm\\Downloads\\Audi_dataset\\"

def run(im,json_old_path,name):
    coords=[]
    #im_disp = im.copy()
    im_draw = im.copy()
    #im_draw2=im.copy()
    # window_name = "Select objects to be tracked here."
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 1800, 1200)
    # cv2.putText(im_draw, img_name+"  ("+str(index_basename)+"/"+str(length_basename)+")", (10, 50),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 255, 255),4)
    # cv2.putText(im_draw, "Press [space] to save; [a] to go back [s=-20]; [d] to go on [w=+20]; [BCKSPACE] to delete ", (10, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),4)

    try:
        with open(json_old_path) as imgdata_old_to_read:
            imgdata_old_to_read_open = json.load(imgdata_old_to_read)
            if imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='' and imgdata_old_to_read_open["Teilez.E3"]=='' and imgdata_old_to_read_open["Teilez.E4"]=='': 
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Elektrik\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Frontend_Motorraum' and imgdata_old_to_read_open["Teilez.E3"]=='' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Elektrik_Frontend_Motoraum\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Kofferraum_Heckklappe' and imgdata_old_to_read_open["Teilez.E3"]=='' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Elektrik_Kofferraum_Heckklappe\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Innenraum' and imgdata_old_to_read_open["Teilez.E3"]=='' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Elektrik_Innenraum\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Frontend_Motorraum' and imgdata_old_to_read_open["Teilez.E3"]=='Scheinwerfer_Blinker' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Elektrik_Scheinwerfer_Blinker\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Türen' and imgdata_old_to_read_open["Teilez.E3"]=='' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Türen\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Türen' and imgdata_old_to_read_open["Teilez.E3"]=='Außenspiegel' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Aussenspiegel\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Kofferraum_Heckklappe' and imgdata_old_to_read_open["Teilez.E3"]=='Heckleuchten_Bremsleuchte' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Elektrik_Heckleuchten_Bremsleuchte\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            elif imgdata_old_to_read_open["Teilez.E1"]=='Elektrik' and imgdata_old_to_read_open["Teilez.E2"]=='Frontend_Motorraum' and imgdata_old_to_read_open["Teilez.E3"]=='Wischeranlage' and imgdata_old_to_read_open["Teilez.E4"]=='':
                try:
                    cv2.imwrite('C:\\Users\\pettm\\Desktop\\Script_data\\Elektrik_FM_Wischeranlage\\'+str(name)+'.jpg',im)
                except:
                    print("DONE!!!")
            # cv2.putText(im_draw, "Sichtweite= " + str(imgdata_old_to_read_open["distance"]) + "m", (10, 600),
            #             cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 20)
            #cv2.putText(im_draw, "labels= " + str(imgdata_old_to_read_open["Teilez.E1"]) +','+ str(imgdata_old_to_read_open["Teilez.E2"]) +','+ str(imgdata_old_to_read_open["Teilez.E3"]) +','+ str(imgdata_old_to_read_open["Teilez.E4"]) +',', (10, 200),
                        #cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            print("Done for every image")
    except:
        print("File " + name + ".json seems to be corrupted")
        pass
 
if __name__ == "__main__":
    basename = []
    index_basename=0

    for root,dirs,files in os.walk(img_json_folder_path,topdown=False):
        for filename in files:
            if filename.endswith((".JPG")):
                #basename.append(os.path.splitext(filename)[0])
                #basename = sorted(basename)
                s=root+"\\"
                d=filename[:-4]
                image_path=s+d+".jpg"
                json_path=s+d+".json"
                #json_new_path=img_json_folder_path+basename[index_basename]+"y.json"
                name=d
                try:
                    im = cv2.imread(image_path)
                    print(image_path)
                except:
                    print("Cannot read image --> exiting.")
                    exit()
                print(name)
                #coords, key = run(im,json_path, basename[index_basename], len(basename), index_basename )
                run(im,json_path,name)

        

            
       

    