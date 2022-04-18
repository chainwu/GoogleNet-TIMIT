#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tgt
import glob
import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import gc
from os.path import exists
import os
import random


# In[2]:

DIRLIST=['TRAIN', 'TEST', 'VALIDATE']
PHLIST=['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
STEP=200
WINSIZE=128
WINSTEP=0.005
DIRBASE="/home/chainwu/GIT/GoogleNet-Phoneme/"
DRATIO=0.8
VRATIO=10
RESZ=256

# In[3]:

def dir_create():
    for d in DIRLIST:
        for p in PHLIST:
            dd = DIRBASE+d+"/"+p
            print("Creating "+dd)
            if not exists(dd):
                os.makedirs(dd)


# In[4]:


def extract_phoneme(i, pslice, phtier):
    stime = i * pslice
    etime = (i+WINSIZE)*pslice
    print("{:.3f}".format(stime), "{:.3f}".format(etime), end=" ")
    plist=phtier.get_annotations_between_timepoints(stime, etime, left_overlap=False, right_overlap=False)
    catlist=[]
    for p in plist:
        #pdur = p.end_time - p.start_time
        if p.text == "":
            #      print("h#", end=" ")
            catlist.append("h#")
        else:
            #       print(p.text, end=" ")
            catlist.append(p.text)
        
    #print()
    mylist = list(dict.fromkeys(catlist))
    print(mylist)


# In[5]:

dir_create()
dirpref=r"/opt/speech-data/TIMIT/"

for d in ['TRAIN', 'TEST']:
    dirpref=r"/opt/speech-data/TIMIT/"+d+"/**/"

    elef = glob.glob(dirpref+"*-spec.png", recursive=True)
    elef.sort()

    for fw in elef:
        if d == 'TRAIN':
            if elef.index(fw) % VRATIO == 0:
                # 10 out of 1
                dplace = 'VALIDATE'
            else:
                dplace = 'TRAIN'
        else:
            dplace = 'TEST'
            
        print(fw, dplace)

        tw=fw[:-9]+'.textgrid'

        tg = tgt.io.read_textgrid(tw, encoding='utf-8', include_empty_intervals=True)
        phtier = tg.get_tier_by_name('Phone')
        ptst = phtier.start_time
        ptet = phtier.end_time
        print(phtier.start_time, phtier.end_time)
    
        img = cv2.imread(fw)
        h, w, ch = img.shape
        #print(h, w)
        tslice = float(ptet-ptst)/w
        for i in range(0, w, STEP):
            #print(i)
            if i + h > w:
                break
            stime = i*tslice
            etime = (i+h)*tslice
            #print(i, "{:.3f}".format(stime), "{:.3f}".format(etime))
            cropimg=img[0:h,i:i+h]
            resizimg = cv2.resize(cropimg, (RESZ,RESZ), interpolation=cv2.INTER_AREA)
            #print("cropping 0:",h, i,":", i+h)
            #print("saving",fw[:-8]+str(i)+".png")
            #print("Time:", etime, stime, (etime-stime)/2)
            ann = phtier.get_annotations_by_time((etime-stime)/2 + stime)
            #plist=phtier.get_annotations_between_timepoints(stime, etime, left_overlap=True, right_overlap=True)
            #print(ann[0].text)
            #catlist=[]
            if ann[0].text == "":
                p = "h#"
            else:
                p = ann[0].text
                                
            fname = fw[29:-8]
            ffname=fname.replace("/","-")
            xname=DIRBASE+dplace+"/"+p+"/"+ffname+str(i)+".png"
            print("Creating", xname)
            #gray_image = cv2.cvtColor(resizimg, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(xname, resizimg)
                                    
            del cropimg
                                    
        del img, resizimg
        #, gray_image
        del tg
        del phtier
        gc.collect()
    
