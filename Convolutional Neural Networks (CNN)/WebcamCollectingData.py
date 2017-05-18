# facerec.py
import cv2, os
#%%
class HaarcascadesFilePaths(object):
    def __init__(self):
        self.fn_haar = 'opencv/haarcascades/haarcascade_frontalface_default.xml'
        self.fn_dir = 'dataset/faces_training_set'
#%%
class FolderNameAndNumberOfSamples(object):     
    def samples_number(self):
        goodInput = False
        while not goodInput:
            try:
                number = int(raw_input("Give the max training sample : ")) 
                if number>0:                    
                    goodInput = True
                    answer = raw_input("Do you confirm for the %d samples for the training y/n :"%number)
                    if ((answer=='y') or (answer=='Y')):
                        return number
                    else:
                        goodInput = False                    
            except ValueError:
                print("That number is not an integer. Try again:")                   
        
    def folder_names(self):
        dirs = HaarcascadesFilePaths()
        fn_haar, fn_dir = dirs.fn_haar, dirs.fn_dir
        folderName = False
        while not folderName:
            try:
                firstLastName = raw_input("Give the full name for the face owner :")
                firstLastName = firstLastName.split()
                if ((not firstLastName) or (len(firstLastName)==1)):
                    folderName = False
                else:
                    fn_name = firstLastName[0]+'-'+firstLastName[1]
                    folderName = True
            except ValueError:
                print("That is not a full name. Try again:") 
        path = os.path.join(fn_dir, fn_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        haar_cascade = cv2.CascadeClassifier(fn_haar)
        webcam = cv2.VideoCapture(0)
        # Generate name for image file
        pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.' ]+[0])[-1] + 1

        return (fn_name, path, haar_cascade, webcam, pin)
#%%                   
class FaceDetection(object):
    def use_webcam(self, count_max, folder_info, img_width, img_height):
        # Get the variables
        self.fn_name, self.path = folder_info[0], folder_info[1] 
        self.haar_cascade, self.webcam = folder_info[2], folder_info[3]
        self.pin, size = folder_info[4], 2
        # Beginning message
        print("\n\033[94mThe program will save 20 samples. \
        Move your head around to increase while it runs.\033[0m\n")
        # The program loops until it has 20 images of the face.
        count, pause = 0, 0
        while count < count_max:
            # Loop until the camera is working
            rval = False
            while(not rval):
                # Put the image from the webcam into 'frame'
                (rval, frame) = self.webcam.read()
                if(not rval):
                    print("Failed to open webcam. Trying again ...")
            # Get image size
            height, width, channels = frame.shape
            # Flip frame
            frame = cv2.flip(frame, 1, 0)
            # Convert image to grayscale instead of having 3 color
            gray = frame # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Scale down for speed
            mini = cv2.resize(gray, (int(gray.shape[1]/size), int(gray.shape[0]/size)))
            # Detect faces
            faces = self.haar_cascade.detectMultiScale(mini)
            # We only consider largest face
            faces = sorted(faces, key = lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * size for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (img_width, img_height))
                # Draw rectangle and write name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, self.fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))
                # Remove false positives
                if(w * 6 < width or h * 6 < height):
                    print("Face too small")
                else:
                    # To create diversity, only save every fith detected image
                    if(pause == 0):
                        print("Saving training sample "+str(count+1)+"/"+str(count_max))
                        # Save image file
                        cv2.imwrite('%s/%s.png' % (self.path, self.pin), face_resize)
                        print 'face_resize', face_resize.shape
                        self.pin += 1
                        count += 1
                        pause = 1                                   
            if(pause > 0):
                pause = (pause + 1) % 5
            cv2.imshow('OpenCV', frame)
            key = cv2.waitKey(5)
            if key == 27:
                break
        # When everything is done, release the capture
        self.webcam.release()
        cv2.destroyAllWindows()
#%%
'''
class MainDataCollectingClass(object):
    def __init__(self, img_width = 200, img_height = 200):
        print "Training...."
        # img_width, img_height = 200, 200
        FolderSamplesInfo = FolderNameAndNumberOfSamples()
        count_max = FolderSamplesInfo.samples_number()
    
        endCollectingData = False
        while not endCollectingData:
            folder_info = FolderSamplesInfo.folder_names()
            Face_track = FaceDetection()
            Face_track.use_webcam(count_max, folder_info, img_width, img_height)
            answer = raw_input ('Dot you want to add Data y/n ? :')
            if (answer!='y'):
                endCollectingData = True
#%%
'''
if __name__=='__main__':
    print "Training...."
    img_width, img_height = 200, 200
    folderSamplesInfo = FolderNameAndNumberOfSamples()
    count_max = folderSamplesInfo.samples_number()
    
    endCollectingData = False
    while not endCollectingData:
        folder_info = folderSamplesInfo.folder_names()
        face_track = FaceDetection()
        train = face_track.use_webcam(count_max, folder_info, img_width, img_height)
        answer =raw_input ('Dot you want to add Data y/n ? :')
        if (answer!='y'):
            endCollectingData = True
            print '---- The Data collection is ended, Thanks -----'

    




