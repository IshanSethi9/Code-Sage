#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, redirect,url_for, request, render_template
import requests
from imutils.video import VideoStream
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import time
import pygame
from pygame import mixer
from flask_ngrok import run_with_ngrok
from scipy.spatial import distance as dist
from imutils.video import FPS
import math

def nothing(x):
        pass

def playGuitar():
    
        cam=cv2.VideoCapture(0)

        mixer.init() 

        time.sleep(2)
        circle_radius = 1

        while True:
            status, frame = cam.read()
            height,width = frame.shape[:2]
            frame = cv2.flip(frame,1);

            frame = imutils.resize(frame, height=300)
            frame = imutils.resize(frame, width=600)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            blueLower = np.array([77,95,42])
            blueUpper = np.array([255,255,255])

            mask = cv2.inRange(hsv, blueLower, blueUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / (M["m00"]+0.000001)), int(M["m01"] / (M["m00"]+0.000001)))
                if radius > circle_radius:
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 0, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

                    if center[0]>50 and center[0]<550 and center[1]>50 and center[1]<75:
                        cv2.putText(frame,'E {Low}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                        pygame.mixer.music.load(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\Music\Open-E-note-low-sixth-string.mp3')
                        pygame.mixer.music.play(0)

                    elif center[0]>50 and center[0]<550 and center[1]>100 and center[1]<125:
                        cv2.putText(frame,'A',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
                        pygame.mixer.music.load(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\Music\Open-A-note-fifth-string.mp3')
                        pygame.mixer.music.play(0)

                    elif center[0]>50 and center[0]<550 and center[1]>150 and center[1]<175:
                        cv2.putText(frame,'D',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                        pygame.mixer.music.load(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\Music\Open-D-note-fourth-string.mp3')
                        pygame.mixer.music.play(0)

                    elif center[0]>50 and center[0]<550 and center[1]>200 and center[1]<225 :
                        cv2.putText(frame,'G',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
                        pygame.mixer.music.load(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\Music\Open-G-note-third-string.mp3')
                        pygame.mixer.music.play(0)

                    elif center[0]>50 and center[0]<550 and center[1]>250 and center[1]<275 :
                        cv2.putText(frame,'B',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
                        pygame.mixer.music.load(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\Music\Open-B-note-second-string.mp3')
                        pygame.mixer.music.play(0)

                    elif center[0]>50 and center[0]<550 and center[1]>300 and center[1]<325:
                        cv2.putText(frame,'E {High}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
                        pygame.mixer.music.load(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\Music\Open-E-note-high-first-string.mp3')
                        pygame.mixer.music.play(0)

            frame_copy=frame.copy()

            frame_copy = cv2.rectangle(frame_copy,(50,50),(550,75),(255,255,255),1)
            cv2.putText(frame_copy,'E {Low}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

            frame_copy = cv2.rectangle(frame_copy,(50,100),(550,125),(0,0,0),1)
            cv2.putText(frame_copy,'A',(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            frame_copy = cv2.rectangle(frame_copy, (50,150),(550,175),(255,255,255),1)
            cv2.putText(frame_copy,'D',(50,150),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

            frame_copy = cv2.rectangle(frame_copy, (50,200),(550,225),(0,0,0),1)
            cv2.putText(frame_copy,'G',(50,200),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            frame_copy = cv2.rectangle(frame_copy, (50,250),(550,275),(255,255,255),1)
            cv2.putText(frame_copy,'B',(50,250),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

            frame_copy = cv2.rectangle(frame_copy, (50,300),(550,325),(0,0,0),1)
            cv2.putText(frame_copy,'E {High}',(50,300),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            cv2.putText(frame_copy,'GUITAR',(150,425),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3)

            cv2.imshow("Frame", frame_copy)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cam.release()
        cv2.destroyAllWindows()
        
def show_images(images):
    for i, img in enumerate(images):
        cv2.imshow("image_" + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def comp(image):
    final_img= str(r"C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\static")+str('\\')+str(image) 
    try:
        img_path=final_img+".png"
        image = cv2.imread(img_path)
        scale_percent = 60 
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (500,500)
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)

        cnts = [x for x in cnts if cv2.contourArea(x) > 100]

        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 2
        pixel_per_cm = dist_in_pixel/dist_in_cm

        for cnt in cnts:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
            wid = euclidean(tl, tr)/pixel_per_cm
            ht = euclidean(tr, br)/pixel_per_cm
            if ht>wid:
                goldenratio = ht/wid
            else :
                goldenratio = wid/ht
            if goldenratio >=1.5 and goldenratio <1.7:
                print("Golden Ratio")
                img2 = cv2.imread(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\static\o7.png')
                scale_percent = 60 
                width = int(img2.shape[1] * scale_percent / 100)
                height = int(img2.shape[0] * scale_percent / 100)
                dim = (500,500)
                img2 = cv2.resize(img2,dim, interpolation = cv2.INTER_AREA)
                dst = cv2.addWeighted(image,0.7,img2,0.5,0)
                cv2.imshow('dst',dst)
            else:
                print("Unsatisfied!")
            cv2.putText(image, "{:.2f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] + 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(image, "{:.2f}cm".format(ht), (int(mid_pt_verticle[0] - 15), int(mid_pt_verticle[1] -15)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(image, "{:.2f}".format(goldenratio),(int(mid_pt_verticle[0]-100), int(mid_pt_verticle[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,5), 3)
            
    except AttributeError:
        #print("Attribute error")
        img_path=final_img+".jpg"    

        image = cv2.imread(img_path)
        scale_percent = 60 
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (500,500)
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        cnts = [x for x in cnts if cv2.contourArea(x) > 100]
        
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 2
        pixel_per_cm = dist_in_pixel/dist_in_cm

        for cnt in cnts:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
            wid = euclidean(tl, tr)/pixel_per_cm
            ht = euclidean(tr, br)/pixel_per_cm
            if ht>wid:
                goldenratio = ht/wid
            else :
                goldenratio = wid/ht
            if goldenratio >=1.5 and goldenratio <=1.7:
                print("Golden Ratio")
                img2 = cv2.imread(r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\static\o7.png')
                scale_percent = 60 
                width = int(img2.shape[1] * scale_percent / 100)
                height = int(img2.shape[0] * scale_percent / 100)
                dim = (500,500)
                img2 = cv2.resize(img2,dim, interpolation = cv2.INTER_AREA)
                dst = cv2.addWeighted(image,0.7,img2,0.5,0)
                cv2.imshow('dst',dst)
            else :
                print("Unsatisfied!")
            cv2.putText(image, "{:.2f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] + 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(image, "{:.2f}cm".format(ht), (int(mid_pt_verticle[0] - 15), int(mid_pt_verticle[1] -15)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(image, "{:.2f}".format(goldenratio),(int(mid_pt_verticle[0]-100), int(mid_pt_verticle[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,5), 3)
    show_images([image])

def liveimage():
    try:
        print("[INFO] Loading model...")
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor", "lorry"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        net = cv2.dnn.readNetFromCaffe(r"C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\OpenCV-Measuring-Object-master\OpenCV-Measuring-Object-master\MobileNetSSD_deploy.prototxt.txt",
                                       r"C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\OpenCV-Measuring-Object-master\OpenCV-Measuring-Object-master\MobileNetSSD_deploy.caffemodel")

        def midpoint(ptA, ptB):
            return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

        print("[INFO] starting video stream...")

        vs = VideoStream(src=0).start()
        unit = "cm"

        time.sleep(2.0)

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=1000)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5 :
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = "{}: {:.2f}%".format(CLASSES[idx],
                        confidence * 100)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            edged = cv2.Canny(gray, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            (cnts, _) = contours.sort_contours(cnts)
            pixelsPerMetric = None
            goldenratio=False

            for c in cnts:
                if cv2.contourArea(c) < 100:
                    continue
                orig = frame.copy()
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
                cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                goldenratio=False
                dimA = (dA  * 0.026458) 
                dimB = (dB * 0.026458) 
                ratio=0
                if dimA>0 and dimB>0:
                    if dimA>dimB:
                        ratio=dimA/dimB
                        if ratio<1.7 and ratio>1.6:
                            goldenratio=True
                    elif dimB>dimA:
                        ratio=dimB/dimA
                        if ratio<1.7 and ratio>1.6:
                            goldenratio=True
                

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(orig, (1000, 1000), (700, 620), (800, 132, 109), -1)
                cv2.putText(orig, "{:.1f}cm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), font,0.65, (255, 0, 255), 2)
                cv2.putText(orig, "{:.1f}cm".format(dimB),(int(trbrX + 10), int(trbrY)), font,0.65, (255, 0, 255), 2)
                if(goldenratio):
                    cv2.putText(orig, '--Golden Ratio--', (700,690),font,0.7,(0xFF, 0xFF, 0x00), 1,font)

            cv2.imshow("Frame", orig)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stream.release() 
        vs.stop()
    except ValueError:
        cv2.destroyAllWindows()
        vs.stream.release() 
        vs.stop()
    except AttributeError:
        cv2.destroyAllWindows()
        vs.stream.release()
        vs.stop()
    except TypeError:
        cv2.destroyAllWindows()
        vs.stream.release()
        vs.stop()
    
app=Flask(__name__,template_folder=r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\templates', static_folder=r'C:\Users\Ishan\Desktop\Hackbenchers#4\Golden-Ratio\static')
run_with_ngrok(app)
@app.route('/strm/<string:strm>')
def strm(strm):
    return render_template('strm.html',strm=strm)

@app.route('/',methods=['POST','GET'])
def MainPage():
    try:
        if request.method == 'POST':
            user=request.form['nm']
            return redirect(url_for('strm',strm=user))
    except AttributeError:
        return render_template('MainPage.html')
    return render_template('MainPage.html')

@app.route('/imageGR',methods=['POST','GET'])
def imageGR():
    try:
        if request.method == 'POST':
            image=request.form['image']
            comp(image)
            return render_template('imageGR.html')        
    except AttributeError:
        return render_template('imageGR.html')
    
    return render_template('imageGR.html')

@app.route('/AboutGR',methods=['POST','GET'])
def AboutGr():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template('AboutGR.html')

@app.route('/human1',methods=['POST','GET'])
def human1():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template('human1.html')

@app.route('/guitar')
def guitar():
    playGuitar()
    return render_template('MainPage.html')

@app.route('/liveGR')
def liveGR():
    try:
        liveimage()
    except AttributeError:
        return render_template('MainPage.html')
    except ZeroDivisionError:
        return render_template('MainPage.html')
    except ValueError:
        return render_template('MainPage.html')
    except TypeError:
        return render_template('MainPage.html')
    return render_template("MainPage.html")

@app.route('/Applications')
def Applications():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Applications.html")

@app.route('/Architecture')
def Architecture():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Architecture.html")

@app.route('/Facial')
def Facial():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Facial.html")

@app.route('/Finance')
def Finance():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Finance.html")

@app.route('/Geometry')
def Geometry():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Geometry.html")

@app.route('/GraphicDesign')
def GraphicDesign():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("GraphicDesign.html")

@app.route('/Nature')
def Nature():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Nature.html")

@app.route('/Photography')
def Photography():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Photography.html")

@app.route('/Quantum')
def Quantum():
    if request.method=='POST':
        user=request.form['nm']
        return redirect(url_for('strm',strm=user))
    return render_template("Quantum.html")

app.run()


# #line 392/271/270/244/178/129/89/84/79/74/69/64 destinations to be changed... 
# #cv2.__version__ = 3.4.9.31
