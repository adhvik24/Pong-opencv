import cv2
import mediapipe as mp
import math


class HandDetector:

    def __init__(self, mode=False, maxHands=2, dconfidence=0.5, tconfidence=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.dconfidence = dconfidence
        self.tconfidence = tconfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.dconfidence,
                                        min_tracking_confidence=self.tconfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.landmarks = []

    def findHands(self, img):

        r_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(r_img)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                hand = {}
                # landmarks
                mylandmarks = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylandmarks.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                box = xmin, ymin, boxW, boxH
                cx, cy = box[0] + (box[2] // 2), \
                    box[1] + (box[3] // 2)

                hand["landmarks"] = mylandmarks
                hand["box"] = box
                hand["center"] = (cx, cy)

                hand["type"] = handType.classification[0].label
                allHands.append(hand)

                self.mpDraw.draw_landmarks(img, handLms,
                                           self.mpHands.HAND_CONNECTIONS)
                cv2.putText(img, hand["type"], (box[0] - 30, box[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
        return allHands, img