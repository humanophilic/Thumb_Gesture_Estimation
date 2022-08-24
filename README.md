# Thumb_Gesture_Estimation
Python Implementation of thumb gesture estimation system

<p align="center">
  <img width="500" height="300" src="https://github.com/humanophilic/Thumb_Gesture_Estimation/blob/main/img/img_1.jpg?raw=true" />
</p>

The system is designed mainly for recognizing the OK sign illustrated in the figure which is considered to be a visible symptom of anterior interosseous nerve palsy caused by parsonage-turner syndrome. Patients often show weakness in opponens pollicis muscle of thumb and flexor digitorum profundus muscle of the  index finger.

# Get started
## Dependencies
Start from building a new conda environment.
```
conda create -n py38 python=3.8
conda activate py38
pip install opencv-python==4.6.0.66
pip install tabulate==0.8.10
```
If you are using apple silicon M1 (M2 not supported):
```
pip install mediapipe-silicon
```
else:
```
pip install mediapipe
```

## Usage
### Camera select
If your device has more than one camera attached, you can select different cameras by modifying `0` of `cap = cv2.VideoCapture(0)` in Line 42 of `main.py`. (e.g., if you have three cameras, `0` stands for the default camera index, and you can modify it into `1` or `2` to select other cameras)

### Run
```
python main.py
```
Follow the instructions (adjust hand gesture for 10s and hold for 5s) and the results will be printed in a table ("O": positive, "X": negative). For the case where only one hand is detected, the results will be shown as left hand.  
   
Note that the results greatly depend on the hand gesture towards the camera, thus, we add 10s （preparation stage） at frontend so that users can adjust hand gesture until circle sign appears at thumb & forefinger. During the next 5s (detection stage), please hold the correct gesture. Final result is only affected by the hand gesture in the last 5s.

# Contact us
For any questions or requests, please contact us at chen.chenhao.419@s.kyushu-u.ac.jp
