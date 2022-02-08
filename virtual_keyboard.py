import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
#from pynput.keyboard import Controller


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)


detector = HandDetector(detectionCon=0.8)
keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",","."," ",],
        ["0","1","2","3","4","5","6","7","8","9"]]
finalText = ""

#keyboard = Controller()


#def drawALL(img, buttonList):
#    for button in buttonList:
#        x,y = button.pos
#        w,h = button.size
#        cv2.rectangle(img, button.pos, (x+w, y+h), (128,0,128), cv2.FILLED)
#        cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
#    return img


def drawALL(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0],button.size[0]), 20 ,rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]), (128,0,128), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)

    out = img.copy()
    alpaha = 0.5
    mask = imgNew.astype(bool)
    #print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpaha, imgNew, 1-alpaha, 0)[mask]
    return out


class Button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text

     
buttonList = []
for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([100*j+50, 100*i+50], key))


while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, draw=False)
    img = drawALL(img, buttonList)

    if lmList:
        for button in buttonList:
            x,y = button.pos
            w,h = button.size

            if x < lmList[4][0] < x+w and y < lmList[4][1] < y+h:                
                cv2.rectangle(img, (x-5,y-5), (x+w+5, y+h+5), (199,21,133), cv2.FILLED)
                cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                l,_,_ = detector.findDistance(4, 8, img, draw=False)
                #print(l)

                #-----when clicked
                if l<30:
                    #keyboard.press(button.text)
                    cv2.rectangle(img, button.pos, (x+w, y+h), (229,204,255), cv2.FILLED)
                    cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                    finalText += button.text
                    sleep(0.25)
                
    cv2.rectangle(img, (50,450), (1035,450), (128,0,128), cv2.FILLED)
    cv2.putText(img, finalText, (55,530), cv2.FONT_HERSHEY_PLAIN, 5, (212,255,127), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
