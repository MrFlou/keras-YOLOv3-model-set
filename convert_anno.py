import glob, os
import chardet
import cv2
import tqdm as tqdm
import math
os.chdir(".\\datasets\\Top_View\\labeled_images")
files = glob.glob(".\\train\\*.txt")
#print(len(files))

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

#'/path/to/000001.jpg': {'100,120,200,235':'dog', '85,63,156,128':'car', ...},
with open('train_anno.txt', 'a') as the_anno:
    #print("Hello?")
    for txt in tqdm.tqdm(files):
        #print("Hello?2")
        txt_file = open(txt,"r")
        boxes = txt_file.readlines()
        image = cv2.imread(txt[:-3]+"jpg")
        shape = image.shape
        outstring = "datasets/Top_View/labeled_images/train/" + txt[8:-3]+"jpg "
        for box in boxes:
            #print("Hello?3")
            item = box.split(" ")
            #print(item)
            centX = shape[1] * float(item[1])
            centY = shape[0] * float(item[2])
            Width = shape[1] * float(item[3])            
            Height = shape[0] * float(item[4])
            minX = int(round_half_up(centX - Width/2))
            minY = int(round_half_up(centY - Height/2))
            maxX = int(round_half_up(centX + Width/2))
            maxY = int(round_half_up(centY + Height/2))
            
            outstring = outstring +str(minX)+","+str(minY)+","+str(maxX)+","+str(maxY)+",14 "
            
            #print(" ")
            #print(shape)
            #print(centX,centY,Width,Height)
            #print(item)
            #print(outstring)
            #print(" ")
            
            #some = input()
        outstring = outstring + "\n"
        #print(outstring)
        the_anno.write(outstring)



            




