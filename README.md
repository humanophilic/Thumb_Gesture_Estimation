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
pip install -r requirements.txt
```
## Usage
```
python main.py
```
Follow the instructions (adjust hand gesture and hold) and the results will be printed in a table ("O": positive, "X": negative). For the case where only one hand is detected, the results will be shown as left hand.

# Contact us
For any questions or requests, please contact us at chen.chenhao.419@s.kyushu-u.ac.jp
