import numpy as np
import cv2
import pygame
import mediapipe as mp
from tracking import HandDetector


pygame.mixer.init()


def imgoverimg(bg, overlimg, pos=[0, 0]):
    hf, wf, _ = overlimg.shape
    hb, wb, cb = bg.shape
    *_, mask = cv2.split(overlimg)
    mask1 = np.zeros((hb, wb, cb), np.uint8)

    b_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    b_mask1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    r_imgA = cv2.bitwise_and(overlimg, b_mask)
    r_img = cv2.cvtColor(r_imgA, cv2.COLOR_BGRA2BGR)

    mask1[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = r_img
    mask2 = np.ones((hb, wb, cb), np.uint8) * 255
    b_img = cv2.bitwise_not(b_mask1)
    mask2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = b_img

    bg = cv2.bitwise_and(bg, mask2)
    bg = cv2.bitwise_or(bg, mask1)

    return bg


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

bg = cv2.imread("img/bg.jpg")
ball = cv2.imread("img/Ball.png", cv2.IMREAD_UNCHANGED)
tile1 = cv2.imread("img/tile1.png", cv2.IMREAD_UNCHANGED)
tile2 = cv2.imread("img/tile1.png", cv2.IMREAD_UNCHANGED)

detector = HandDetector(dconfidence=0.8, maxHands=2)

position = [100, 100]
vx = 15
vy = 15
endgame = False
score = [0, 0]

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    hands, img = detector.findHands(img)

    img = cv2.addWeighted(img, 0.2, bg, 0.8, 0)

    if hands:
        for hand in hands:
            x, y, w, h = hand['box']
            h1, w1, _ = tile1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = imgoverimg(img, tile1, (59, y1))
                if 59 < position[0] < 59 + w1 and y1 < position[1] < y1 + h1:
                    vx = -vx
                    position[0] += 30
                    score[0] += 1
                    pygame.mixer.music.load("img/sound_bounce.ogg")
                    pygame.mixer.music.play()

            if hand['type'] == "Right":
                img = imgoverimg(img, tile2, (1195, y1))
                if 1195 - 50 < position[0] < 1195 and y1 < position[1] < y1 + h1:
                    vx = -vx
                    position[0] -= 30
                    score[1] += 1
                    pygame.mixer.music.load("img/sound_bounce.ogg")
                    pygame.mixer.music.play()

    if position[0] < 40 or position[0] > 1200:
        endgame = True
        if(position[0] < 40 or position[0] > 1200):
            pygame.mixer.music.load("img/out.wav")
            pygame.mixer.music.play()
            position = [100, 100]

    if endgame:
        bg1 = bg.copy()
        img = bg1
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    3, (0, 200, 0), 5)
        cv2.putText(img, str("Your score"), (380, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 5)
        cv2.putText(img, str("q-Quit"), (800, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 0, 255), 2)
        cv2.putText(img, str(" r-Regame"), (150, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 255, 0), 2)

    else:

        if position[1] >= 500 or position[1] <= 10:
            vy = -vy
            pygame.mixer.music.load("img/sound_bounce.ogg")
            pygame.mixer.music.play()

        position[0] += vx
        position[1] += vy

        img = imgoverimg(img, ball, position)

        cv2.putText(img, str(score[0]), (300, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 5)
        cv2.putText(img, str(score[1]), (900, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        position = [100, 100]
        vx = 15
        vy = 15
        endgame = False
        score = [0, 0]
        completed = cv2.imread("img/bg.jpg")
    if key == ord('q'):
        break
