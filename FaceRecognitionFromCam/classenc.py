#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  classenc.py
#   
# 
 
import cvlib as cvl
import cv2
import face_recognition
import os
import pickle

def enclass(path, save):
    os.chdir(path)
    classenc = []
    dlist = os.scandir('.')
    for dent in dlist:
        if not dent.is_dir():
            continue
        name = dent.name
        ilist = os.scandir(name)
        for ient in ilist:
            iname = ient.name
            print(name)
            img = cv2.imread(name+"/"+iname)
            print(img.shape)
            face, conf = cvl.detect_face(img)
            if len(conf) < 1:
                continue
            print(conf[0], face[0])
            
            enc = face_recognition.face_encodings(img, face)
            obj = name, enc
            classenc.append(obj)
            
    pickle.dump(classenc, open(save, "wb"))
    
    return 0
    
def declass(read):
    classenc = pickle.load( open(read, "rb"))
    for name,enc in classenc:
        print(name, len(enc))
    
def main(args):
    enclass('images', 'class.enc')
    declass('class.enc')
    
    return 0
        
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
