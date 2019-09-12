# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:42:21 2017

@author: omerf
"""

import cv2
import numpy as np
import sys
import time
from shapedetector import ShapeDetector

sd = ShapeDetector()

fps = 0

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

if cap.isOpened():
    print('Kamera Açıldı')
else:
    print('HATA!! \nKamera Açılamadı!!')
    exit(1)

frame_count = 0
while 1:

    try:
        start = time.time()
        ret, frame = cap.read()

        if not ret:
            print('HATA!! Frame Alınamıyor \nYeniden Başlatın')
            cv2.destroyAllWindows()
            cap.release()
            break

        fgmask = fgbg.apply(frame)

        (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            shape = sd.detect(c)
            print(shape)
            approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)

            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)
            # print(len(approx))
            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, "Fps: "+str(fps), (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        fgmask = cv2.resize(fgmask, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('fgmask', fgmask)
        cv2.imshow('frame', frame)
        frame_count += 1
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            print("Çıkış Yapıldı")
            break

        fps = np.float16((1 / (time.time() - start)))

    except:
        print("Beklenmedik Hata!!! ", sys.exc_info()[0])
        raise

cv2.destroyAllWindows()
cap.release()