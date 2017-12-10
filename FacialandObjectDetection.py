import sensor, time, image, pyb, gc
import os


# Reset sensor
sensor.reset()

# Sensor settings
sensor.set_contrast(1)
sensor.set_gainceiling(16)
# HQVGA and GRAYSCALE are the best for face tracking.
sensor.set_framesize(sensor.QQVGA)
sensor.set_pixformat(sensor.GRAYSCALE)

#gc.enable()
# Load Haar Cascade
# By default this will use all stages, lower satges is faster but less accurate.
face_cascade = image.HaarCascade("frontalface", stages=25)
print(face_cascade)

# FPS clock
clock = time.clock()

#RedLED object
red_led = pyb.LED(1)

#GreenLED object
green_led = pyb.LED(2)

#BlueLED object
blue_led = pyb.LED(3)

NumOfFaces = 1
distance = 0
samePerson = 0
firstRunForFace = 0

#Make directory for tempSnapshot for comparison
os.mkdir("snapshot")

#first function called in program to check if sd card is empty
def SDCardEmpty():
        if not "1" in os.listdir():
            #NumOfFaces += 1
            print("dir #")
            print("1")
            os.mkdir("%d" % (1))
            #stores 9 images on the sd card in folder NumOfFaces
            for x in range (0,9):
                sensor.snapshot().save("%d/snapshot-%d.pgm" % (1, x))

#Call if there is a face found in frame
def StoreSnapshot(img):
        img.save("snapshot/snap.pgm")
        img = image.Image("snapshot/snap.pgm",copy_to_fb = True).mask_ellipse()
        print("Retreived Image from Snapshot")
        #Calculate the LBP of the snapshot image
        imgSnapSimilarity = img.find_lbp((0, 0, img.width(), img.height()))
        return imgSnapSimilarity

#Find Face match from sd card
def FindFaceMatch(imgSnapSimilarity):
    global distance
    global NumOfFaces
    for j in range(1,NumOfFaces+1):
        distance = 0
        for i in range(0,9):
            #Grab file from SD Card to compare to
            img = image.Image("%d/snapshot-%d.pgm"%(j,i),copy_to_fb = True).mask_ellipse()
            #Calculate the LBP of the saved image
            savedImg = img.find_lbp((0, 0, img.width(), img.height()))
            #calculate distance of snapshot image vs the images saved
            #90 is perfect threshold for detection
            distance += image.match_descriptor(imgSnapSimilarity, savedImg, 90)
            print("File %d Distance %d:" %(j,i))
            savedImg = None
            img = None
        gc.collect()
        if(distance < 350000):
            print(distance)
            return distance
        print(distance)
    print("Not Recognized")
    return distance

#capture Face that I dont recognize and store it
def captureNewFace():
    global NumOfFaces
    NumOfFaces += 1
    print("dir #")
    print(NumOfFaces)
    os.mkdir("%d" % (NumOfFaces))
    #stores 9 images on the sd card in folder NumOfFaces
    for x in range (0,9):
        sensor.snapshot().save("%d/snapshot-%d.pgm" % (NumOfFaces, x))

#Main function for detection of faces and objects
while (True):
    clock.tick()
    img = sensor.snapshot()

    #Find face objects in snapshot
    objects = img.find_features(face_cascade, threshold=0.75, scale_factor=1.25)

    #Find black rectangles in snapshot
    rectobj = img.find_rects(threshold = 10000)

    #if any Faces are found run through the cycle
    if(any(objects)):
        if(firstRunForFace == 0):
            SDCardEmpty()
            firstRunForFace = 1

        gc.collect()
        print("free mem")
        print(gc.mem_free())
        print("snapshot directory")
        imgSnapSimilarity = StoreSnapshot(img)
        img = None
        distance = FindFaceMatch(imgSnapSimilarity)
        imgSnapSimilarity = None

        #Tested and 350000 is the good metric value to compare against for a set of images
        if(distance < 350000):
            blue_led.off()
            red_led.off()
            green_led.on()
            print("Face Recognized")
            distance = 0
        else:
            blue_led.off()
            green_led.off()
            red_led.on()
            print("Face not Recognized")
            captureNewFace()
    elif(any(rectobj)):
        #rectobj = img.find_rects(threshold = 10000)
        for r in rectobj:
            img.draw_rectangle(r.rect(), color = (255, 0, 0))
            #Turn on LED red for object detection
            green_led.off()
            red_led.on()
            blue_led.on()
            print("Detecting Rectangle Object")
    else:
        green_led.off()
        red_led.off()
        blue_led.off()

    print(clock.fps())
