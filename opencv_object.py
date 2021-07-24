    # python opencv_object.py --video test.mp4 --tracker csrt

    # import the necessary packages
from imutils.video import VideoStream##Ehsan:just for using webam
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
#from cydets.algorithm import detect_cycles
import ctypes
import os
import rainflow
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import scipy.optimize as op
from scipy import fftpack
from scipy import signal
from scipy import optimize

def camera1():
    
    
    def Mbox(title, text, style):
        return ctypes.windll.user32.MessageBoxW(0, text, title, style)

    def round_half_up(n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n*multiplier + 0.5) / multiplier

    class Plotter:
        def __init__(self, plot_width, plot_height,sample_buffer=None):
            self.width = plot_width
            self.height = plot_height
            self.color = (255, 0 ,0)
            self.plot_canvas = np.ones((self.height, self.width, 3))*255    
            self.ltime = 0
            self.plots = {}
            self.plot_t_last = {}
            self.margin_l = 10
            self.margin_r = 10
            self.margin_u = 10
            self.margin_d = 50
            self.sample_buffer = self.width if sample_buffer is None else sample_buffer


        # Update new values in plot
        def plot(self, val, label = "plot"):
            if not label in self.plots:
                self.plots[label] = []
                self.plot_t_last[label] = 0
                
            self.plots[label].append(int(val))
            
            while len(self.plots[label]) > self.sample_buffer:
                self.plots[label].pop(0)
                self.show_plot(label)
       
            # Show plot using opencv imshow
        def show_plot(self, label):

            self.plot_canvas = np.zeros((self.height, self.width, 3))*255
            cv2.line(self.plot_canvas, (self.margin_l, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u ), (self.width-self.margin_r, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u), (0,0,255), 1)        

            # Scaling the graph in y within buffer
            scale_h_max = max(self.plots[label])
            scale_h_min = min(self.plots[label]) 
            scale_h_min = -scale_h_min if scale_h_min<0 else scale_h_min
            scale_h = scale_h_max if scale_h_max > scale_h_min else scale_h_min
            scale_h = ((self.height-self.margin_d-self.margin_u)/2)/scale_h if not scale_h == 0 else 0
            

            for j,i in enumerate(np.linspace(0,self.sample_buffer-2,self.width-self.margin_l-self.margin_r)):
                i = int(i)
                cv2.line(self.plot_canvas, (j+self.margin_l, int((self.height-self.margin_d-self.margin_u)/2 +self.margin_u- self.plots[label][i]*scale_h)), (j+self.margin_l, int((self.height-self.margin_d-self.margin_u)/2  +self.margin_u- self.plots[label][i+1]*scale_h)), self.color, 1)
            
            
            cv2.rectangle(self.plot_canvas, (self.margin_l,self.margin_u), (self.width-self.margin_r,self.height-self.margin_d), (255,255,255), 1) 
            cv2.putText(self.plot_canvas,f" {label} : {self.plots[label][-1]} , dt : {int((time.time() - self.plot_t_last[label])*1000)}ms",(int(0),self.height-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.circle(self.plot_canvas, (self.width-self.margin_r, int(self.margin_u + (self.height-self.margin_d-self.margin_u)/2 - self.plots[label][-1]*scale_h)), 2, (0,200,200), -1)
            
            self.plot_t_last[label] = time.time()
            cv2.imshow(label, self.plot_canvas)
            cv2.waitKey(1)

    def normalize_array(array):
        #** for array
        average_of_array = np.mean(array)
        std_dev = np.std(array)

        for i in range(len(array)):
            array[i] = ((array[i] - average_of_array)/std_dev)
        return array


    #Cross-Correlation algorithm
    def get_similarity(image1, image2):
        gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        mat1 = np.asfarray(gray_image, dtype='float').flatten()
        mat2 = np.asfarray(gray_image2, dtype='float').flatten()
       
        rSum = 0.0
        rSum1 = 0.0
        rSum2 = 0.0

        for i in range(len(mat1)):
            rSum += (mat1[i]*mat2[i])**2
            rSum1 += (mat1[i])**2
            rSum2 += (mat2[i])**2
        
        rSum /= (rSum1 * rSum2)**0.5
        
        return rSum

    def findPeak(signal_arr):
        dary = signal_arr
        dary -= np.average(dary)

        step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))

        dary_step = np.convolve(dary, step, mode='valid')

        # Get the peaks of the convolution
        peaks = signal.find_peaks(dary_step, width=2)[0]
        return peaks


    #-----------------------------------------------main()-----------------------------------------------------#
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]

    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(args["tracker"].upper())
    else: # initialize a dictionary that maps strings to their corresponding OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mil": cv2.TrackerMIL_create,
        }
        # grab the appropriate object tracker using our dictionary of OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        tracker2 = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

    # initialize the bounding box coordinates of the object we are going to track
    initBB = None
    initBB2 = None

    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    else: # otherwise, grab a reference to the video file
        vs = cv2.VideoCapture(args["video"])

    # initialize the FPS throughput estimator
    fps = None
    fps_video = vs.get(cv2.CAP_PROP_FPS)

    xlist = list()
    xlist2 = list()
    ylist = list()
    ylist2 = list()
    frameNo = 0

    index = 0
    indexFrm = 0
    v = 0
    # loop over frames from the video stream
    p = Plotter(200, 200,sample_buffer=200)
    first_Image = None
    maxcl = 0
    maxcl2 = 0
    pplotpix = []
    pplotpix2 = []
    index_per_cycle = 0
    index_per_cycle2 = 0
    rate_per_ten = list()
    rate_per_ten2 = list()
    heart_rate_realtime = list()
    heart_rate_realtime2 = list()
    select_index=0#Hossein
    while True:
        # grab the current frame, then handle if we are using a VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame

        # check to see if we have reached the end of the stream
        if frame is None:
            break

        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=1000)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = (1, initBB)
            (success2, box2) = (1, initBB2)

            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                crop_img = frame[y:y + h, x:x + w]
                
                (x2, y2, w2, h2) = [int(v2) for v2 in box2]
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                crop_img2 = frame[y2:y2 + h2, x2:x2 + w2]
     
                if(int(index) == 0):
                    first_Image = crop_img
                    first_Image2 = crop_img2
                    
                index += 1
                cv2.imshow("cropped field", crop_img)
                cv2.imshow("cropped field2", crop_img2)
                
                if index < fps_video * 1:
                    gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    gray_image2 = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2GRAY)
                    Binary = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                    Binary2 = cv2.adaptiveThreshold(gray_image2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                    cl = np.average(Binary)
                    cl2 = np.average(Binary2)
                    
                    if cl > maxcl:
                        maxcl = cl
                        first_Image = crop_img
                        text = "{}: {}".format(index+1, cl)
                        cv2.putText(crop_img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if cl2 > maxcl2:
                        maxcl2 = cl2
                        first_Image2 = crop_img2
                        text2 = "{}: {}".format(index+1, cl2)
                        cv2.putText(crop_img2, text2, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                else:
                    cl = get_similarity(first_Image,crop_img)
                    cl2 = get_similarity(first_Image2,crop_img2)
                    pplotpix.append(cl)
                    pplotpix2.append(cl2)
                    p.plot(cl,label='result')
                    p.plot(cl2,label='result2')
                    indexFrm += 1
                    
                    if indexFrm >= fps_video * 2 and indexFrm % fps_video == 0:
                        y_data = np.array(pplotpix)
                        x_data = np.arange(0,len(y_data))
                        y_data = normalize_array(y_data)
                        s = UnivariateSpline(x_data, y_data, s=50)
                        yy_data = s(x_data)
                        peaks = findPeak(yy_data)
                        
                        y_data2 = np.array(pplotpix2)
                        x_data2 = np.arange(0,len(y_data2))
                        y_data2 = normalize_array(y_data2)
                        s2 = UnivariateSpline(x_data2, y_data2, s=50)
                        yy_data2 = s2(x_data2)
                        peaks2 = findPeak(yy_data2)
                        
                        cycle = 0
                        heatrate = 0
                        if(len(peaks) > 1):
                            for ii in range(len(peaks)):
                                if ii < len(peaks)-1 :
                                    cycle += (peaks[ii+1]-peaks[ii])
                            cycle  = cycle/(len(peaks)-1)
                            cycle = cycle/fps_video
                            heatrate = 60/ cycle
                            heatrate *= 6

                        if(heatrate != 0):
                            index_per_cycle += 1
                            # if index_per_cycle % 10 == 0:
                            rate_per_ten.append(heatrate)
                            heart_rate_realtime.append(heatrate)
                        
                            print("Heart rate=>"+str(heatrate))
                            
                        cycle2 = 0
                        heatrate2 = 0
                        if(len(peaks2) > 1):
                            for ii in range(len(peaks2)):
                                if ii < len(peaks2)-1:
                                    cycle2 += (peaks2[ii+1]-peaks2[ii])
                            cycle2  = cycle2/(len(peaks2)-1)
                            cycle2 = cycle2/fps_video
                            heatrate2 = 60/ cycle2
                            heatrate2 *= 6

                        if(heatrate2 != 0):
                            index_per_cycle2 += 1
                            # if index_per_cycle % 10 == 0:
                            rate_per_ten2.append(heatrate2)
                            heart_rate_realtime2.append(heatrate2)
                        
                            print("Heart rate 2=>"+str(heatrate2))   
                            
                            
            area = (w - x) * (h - y)
            area2 = (w2 - x2) * (h2 - y2)
            frameNo += 1
            xlist.append(frameNo)
            ylist.append(area)
            xlist2.append(frameNo)
            ylist2.append(area2)

            fps.update()
            fps.stop()

            # initialize the set of information we'll be displaying on the frame
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF  # k = cv2.waitKey(1) & 0xff followed by if k == 27 : break--> leads to Exit (oxff--> ESC pressed). this line only initiate "key" and nothing happens by pressing "Esc"comment by Ehsan.

        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        
        if key == ord("s") and select_index==0:
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
            Mbox("Hint message", "Please press S and select the second region, then press Enter.", 0)
            initBB2 = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
            select_index=1#Hossein
            tracker.init(frame, initBB)
            tracker2.init(frame, initBB2)
            fps = FPS().start()
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

    y_data = np.array(pplotpix)
    x_data = np.arange(0,len(y_data))
    y_data = normalize_array(y_data)
    s = UnivariateSpline(x_data, y_data, s=50)
    yy_data = s(x_data)
    peaks = findPeak(yy_data)
    
    y_data2 = np.array(pplotpix2)
    x_data2 = np.arange(0,len(y_data2))
    y_data2 = normalize_array(y_data2)
    s2 = UnivariateSpline(x_data2, y_data2, s=50)
    yy_data2 = s2(x_data2)
    peaks2 = findPeak(yy_data2)

    heartrate = 0
    for i in range(0,index_per_cycle):
        heartrate += rate_per_ten[i]

    heartrate = heartrate / index_per_cycle
    print("total_Heart rate=>"+str(heartrate))
    
    
    heartrate2 = 0
    for i in range(0,index_per_cycle2):
        heartrate2 += rate_per_ten2[i]

    heartrate2 = heartrate2 / index_per_cycle2
    print("total_Heart rate 2=>"+str(heartrate2))

    #heartrate = round_half_up(heartrate, 0)
    if(heartrate < 10):
        heartrate = heart_rate_realtime[-1]
    if(heartrate2 < 10):
        heartrate2 = heart_rate_realtime2[-1]

    Mbox("Measure result", "Total heartbeat of this animal is " + str(heartrate), 0)
    Mbox("Measure result 2", "Total heartbeat of this animal 2 is " + str(heartrate2), 0)
    plt.figure()
    plt.plot(yy_data)    
    # for ii in range(len(peaks)):
        # plt.plot((peaks[ii], peaks[ii]), (-3, 3), 'r')
    # plt.show()
    
    plt.plot(yy_data2)    
    plt.show()

    data_save = {'Heartbeat every 10 seconds': heart_rate_realtime, 'Average heart rate(/min)': heartrate}
    df_save = pd.DataFrame(data = data_save)
    df_save.to_csv('result_bps.csv')
    
    data_save2 = {'Heartbeat 2 every 10 seconds': heart_rate_realtime2, 'Average heart rate 2(/min)': heartrate2}
    df_save2 = pd.DataFrame(data = data_save2)
    df_save2.to_csv('result_bps2.csv')

    Mbox("Success message", "Result saved to 'result_bps.csv' successfully!", 0)
    Mbox("Success message", "Result 2 saved to 'result_bps2.csv' successfully!", 0)
        #os.system("Excel.exe" + 'result_bps.csv')
        #os.system("Excel.exe" + 'result_bps2.csv')
        #close all windows
        #cv2.destroyAllWindows()

    return yy_data,yy_data2
 
    
if __name__=="__main__":
    
    def Mbox(title, text, style):
            return ctypes.windll.user32.MessageBoxW(0, text, title, style)
    Mbox("Hint message", "Please press S while playing the video and select the first region and press Enter, then select the second region with the same procedure.", 0)
    [yy_data,yy_data2]=camera1()

 
    


    
