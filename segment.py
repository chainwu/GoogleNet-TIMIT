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
STEP=100
WINSIZE=128
WINSTEP=0.005
DIRBASE="/home/chainwu/GIT/GoogleNet-Phoneme/"
DRATIO=0.8
VRATIO=10


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
            if i + h > w:
                break
            stime = i*tslice
            etime = (i+h)*tslice
            #print(i, "{:.3f}".format(stime), "{:.3f}".format(etime))
            cropimg=img[0:h,i:i+h]
            #print("cropping 0:",h, i,":", i+h)
            #print("saving",fw[:-8]+str(i)+".png")
            plist=phtier.get_annotations_between_timepoints(stime, etime, left_overlap=True, right_overlap=True)
            catlist=[]
            for p in plist:
                if p.start_time < stime and p.end_time > etime:
                    pinclude = True
                elif p.start_time < stime or p.end_time > etime:
                    # Phoneme is partial overlapped, calculate the portion inside
                    pdur = p.end_time - p.start_time
                    
                    dstime = max(p.start_time, stime)
                    detime = min(p.end_time, etime)
                    
                    if ((detime - dstime) / pdur) < DRATIO:
                        pinclude = False
                    else:
                        pinclude = True
                else:
                    pinclude = True
                        
                if pinclude == True:
                    if p.text == "":
                        catlist.append("h#")
                    else:
                        catlist.append(p.text)
                                
            mylist = list(dict.fromkeys(catlist))
            #print(mylist)

            for p in mylist:
                fname = fw[29:-8]
                ffname=fname.replace("/","-")
                xname=DIRBASE+dplace+"/"+p+"/"+ffname+str(i)+".png"
                print("Creating", xname)
                gray_image = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(xname, gray_image)
                                    
            del cropimg
            del mylist
            del plist
            del catlist
                                    
        del img
        del tg
        del phtier
        gc.collect()
    
