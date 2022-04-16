#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tgt
import glob
import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from PIL import Image
import cv2
import gc
import math

# In[2]:

DATADIR=['TRAIN', 'TEST']
PHLIST=['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
STEP=100
WINSIZE=128
WINSTEP=0.005
MYDPI=100
MAXFREQ=8000
FREQSTEP=80
# In[3]:


def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='gray_r')
    #plt.ylim([spectrogram.ymin, spectrogram.ymax])
    #plt.xlabel("time [s]")
    #plt.ylabel("frequency [Hz]")
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')
    del X
    del Y
    del sg_db
    gc.collect()


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

for d in DATADIR:
    dirpref=r"/opt/speech-data/TIMIT/"+d+"/**/"
    elef = glob.glob(dirpref+"*.wav", recursive=True)
    elef.sort()
    for w in elef:
        print(w)
        tw=w[:-3]+'textgrid'

        snd = parselmouth.Sound(w)
        tg = tgt.io.read_textgrid(tw, encoding='utf-8', include_empty_intervals=True)
        phtier = tg.get_tier_by_name('Phone')
        ptst = phtier.start_time
        ptet = phtier.end_time
        print(phtier.start_time, phtier.end_time)
    
        winstart = 0
        #plt.figure(figsize=(20,4))
        #plt.plot(snd.xs(), snd.values.T)
        #plt.xlim([snd.xmin, snd.xmax])
        #plt.xlabel("time [s]")
        #plt.ylabel("amplitude")
        #plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

        #intensity = snd.to_intensity()
        spectrogram = snd.to_spectrogram(maximum_frequency=MAXFREQ,frequency_step=FREQSTEP)
        draw_spectrogram(spectrogram)

        figure = plt.gcf()
        figure.set_size_inches(math.ceil(10*(ptet-ptst)), 4)
        plt.gray()
        plt.savefig(w[:-4]+"-spec.png", bbox_inches='tight',pad_inches = 0, dpi=MYDPI)
        #plt.show()
        plt.close()
        #plt.clf()
    
        #s=spectrogram.as_array()
        #print(s.shape)
        #finshape=s.shape[-1]
        #pslice = (phtier.end_time - phtier.start_time)/finshape
        #for i in range(0,finshape, STEP):
        #    if i+WINSIZE > finshape:
        #        break
        #    else:
        #        sx=s[:,i:i+WINSIZE]
        #        print(i,sx.shape)
        #print(sx)
        #extract_phoneme(i, pslice, phtier)
        #im = Image.fromarray(sx)
        #im = im.convert("RGB")
        #im.save(w[:-3]+"-"+str(i)+"-"+str(i+WINSIZE)+".png")
        
        #sx= s[:, finshape-WINSIZE:finshape]
        #print(i, sx.shape, end=" ")
        #extract_phoneme(i, pslice, phtier)
        #print()
    
        del snd
        del tg
        del spectrogram
        #del s
        #del phtier
        #del sx
        gc.collect()
    # End of processing




