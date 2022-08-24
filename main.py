import mediapipe as mp
import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import math
import shutil
import time
from tabulate import tabulate

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DP = 2.4
MIN_DIST = 40

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2) 
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 
        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d     
        return int(x3), int(y3), int(x4), int(y4)

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        frame_th = 0
        _hand_det, hand_det = 0, 0
        time_start, time_current = 0, 0
        hold = 0
        left_score, right_score = [], []
        while cap.isOpened():
            # read frame
            success, image = cap.read()
            frame_th += 1
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            output = image.copy()
            
            # visualize landmarks
            if results.multi_hand_landmarks:
                hand_det = 1
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        output,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                output = cv2.flip(output, 1)
                cv2.putText(output, 'Hands found: Yes', (40,60), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0,252,124), 2)
                if _hand_det==0:
                    time_start = time.time()
                    cv2.putText(output, 'Please adjust hand gesture in 10s: 10s', (40,110), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0,252,124), 2)
                elif _hand_det==1:
                    time_current = time.time()
                    if int(time_current-time_start)<10:
                        cv2.putText(output, 'Please adjust hand gesture in 10s: {}s'.format(10-int(time_current-time_start)), (40,110), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0,252,124), 2)
                    elif int(time_current-time_start)==10:
                        cv2.putText(output, 'Please hold for 5s: 5s', (40,110), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0,252,124), 2)
                    else:
                        if int(time_current-time_start)<=15:
                            hold = 1
                            cv2.putText(output, 'Please hold for 5s: {}s'.format(15-int(time_current-time_start)), (40,110), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0,252,124), 2)
                        else:
                            hold = 0
                            if len(right_score)==0 and len(left_score)>0:
                                print('\nHand number detected: 1')
                                left_state = 'X' if sum(left_score)/len(left_score)<0.25 else 'O'
                                data = {
                                    'Left hand': [left_state],
                                    'Right hand': ['-']
                                }
                                print('')
                                print(tabulate(data, headers='keys', stralign='center'), '\n')
                            elif len(right_score)>0 and len(left_score)>0:
                                print('\nHand number detected: 2')
                                left_state = 'X' if sum(left_score)/len(left_score)<0.25 else 'O'
                                right_state = 'X' if sum(right_score)/len(right_score)<0.25 else 'O'
                                data = {
                                    'Left hand': [left_state],
                                    'Right hand': [right_state]
                                } 
                                print('')                               
                                print(tabulate(data, headers='keys', stralign='center'), '\n')
                            break
            else:
                hand_det = 0
                output = cv2.flip(output, 1)
                cv2.putText(output, 'Hands found: No', (40,60), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (13,23,227), 2)

            output = cv2.flip(output, 1)
            cv2.imshow('MediaPipe Hands', cv2.flip(output, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            if not results.multi_hand_landmarks:
                _hand_det = hand_det
                continue

            # extract landmark
            _hand_det = hand_det
            image_height, image_width, _ = image.shape
            hand_loca = [[] for _ in range(len(results.multi_hand_landmarks))]
            for i in range(len(results.multi_hand_landmarks)):
                hand_loca[i].append(results.multi_hand_landmarks[i].landmark[9].x)
                # compare distance
                dist = []
                for j in [[4,8], [8,7]]:
                    d_x1 = results.multi_hand_landmarks[i].landmark[j[0]].x * image_width
                    d_y1 = results.multi_hand_landmarks[i].landmark[j[0]].y * image_height
                    d_x2 = results.multi_hand_landmarks[i].landmark[j[1]].x * image_width
                    d_y2 = results.multi_hand_landmarks[i].landmark[j[1]].y * image_height
                    d = math.sqrt(math.pow((d_x1-d_x2),2) + math.pow((d_y1-d_y2),2))
                    dist.append(d)
                if dist[0]*1.5>dist[1]:
                    hand_loca[i].append(0)
                    continue

                coords = []
                for j in [2,3,4,8,7,6]:
                    rela_x = results.multi_hand_landmarks[i].landmark[j].x
                    rela_y = results.multi_hand_landmarks[i].landmark[j].y
                    x = int(rela_x * image_width)
                    y = int(rela_y * image_height)
                    coords.append((x, y))
                
                fig = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                for k in range(5):
                    cv2.line(fig, coords[k], coords[k+1], (255,255,255), 5)

                # find symmetry point
                sym_coords = []
                for k in range(1,5):
                    x0, y0 = coords[0]
                    x1, y1 = coords[-1]
                    x3, y3 = coords[k]
                    r0 = math.sqrt(math.pow((x3-x0),2) + math.pow((y3-y0),2))
                    r1 = math.sqrt(math.pow((x3-x1),2) + math.pow((y3-y1),2))
                    x4, y4, x5, y5 = get_intersections(x0, y0, r0, x1, y1, r1)
                    if x4==x3:
                        sym_coords.append((x5, y5))
                    else:
                        sym_coords.append((x4, y4))     
                for k in range(3):
                    cv2.line(fig, sym_coords[k], sym_coords[k+1], (255,255,255), 5)
                cv2.line(fig, sym_coords[0], coords[0], (255,255,255), 5)
                cv2.line(fig, sym_coords[-1], coords[-1], (255,255,255), 5)
                cv2.imwrite('./tmp/tmp_{}_{}.jpg'.format(str(frame_th).rjust(5,'0'), str(i)), fig)

                # circle detection
                img = cv2.imread('./tmp/tmp_{}_{}.jpg'.format(str(frame_th).rjust(5,'0'), str(i)))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, DP, MIN_DIST)
                if circles is not None:
                    hand_loca[i].append(1)
                    circlesRound = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circlesRound:
                        cv2.circle(output, (x, y), r, (0,215,255), 5)
                else:
                    hand_loca[i].append(0)
                
            if hold==1:
                if len(hand_loca)>1:
                    if hand_loca[1][0]<=hand_loca[0][0]:
                        left_score.append(hand_loca[0][1])
                        right_score.append(hand_loca[1][1])
                    else:
                        right_score.append(hand_loca[0][1])
                        left_score.append(hand_loca[1][1])
                else:
                    left_score.append(hand_loca[0][1])
            else:
                left_score, right_score = [], []

            cv2.imshow('MediaPipe Hands', cv2.flip(output, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__=='__main__': 
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.mkdir('./tmp')
    main()