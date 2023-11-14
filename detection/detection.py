import cv2 
import numpy as np
import model
import glob
from numpy.linalg import norm

class detect_face:
    def __init__(self):

        self.model = model.model()
    
    
    def extract_image(self,path,name):
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            new_img = img[y:y+h, x:x+w]
            vector = self.model.extract(cv2.resize(new_img,(224,224)))
        np.savetxt("./data/" +name,vector)
    
    def detect(self):
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        
        while (True):
            ret, img = cap.read()
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            new_img = None
           
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                # roi_gray = gray[y:y+h, x:x+w]
                # roi_color = img[y:y+h, x:x+w]
                new_img = img[y:y+h, x:x+w]
                vector = self.model.extract(cv2.resize(new_img,(224,224)))

                name = "unknow"
                
                score = 0
                for i in sorted(glob.glob('./data/*')):
                  
                    temp_score = np.dot(vector, np.loadtxt(i)) / (norm(vector) * norm(np.loadtxt(i)))
                    print(temp_score)
                    if temp_score > score and temp_score>0.7:
                        score = temp_score
                        name = str(i) + " " + str(score)

            
                cv2.putText(img,name,(x, y + int(35 * 0.5)),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),thickness = 1)    
            cv2.imshow('img',img)

    

            
            if cv2.waitKey(1) & 0xFF == 27:
                if new_img is not None and name == "unknow":
                    person = input()
                    vector = self.model.extract(cv2.resize(new_img,(224,224)))
                    np.savetxt("./data/" +person,vector)
                #break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
                


######################################

detect = detect_face()
#detect.extract_image('./image/Khoa.jpg',"Khoa")
detect.detect()

        
    
# for i in sorted(glob.glob('./data/*')):
#     a = np.load(i)
# print(a["Ngoc"])