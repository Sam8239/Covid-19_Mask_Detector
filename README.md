
# MASK DETECTION USING HAAR FEATURE-BASED CASCADE CLASSIFIER

Haar is a machine learning based cascade classifier. 
It is an effective object detection approach in which a cascade function is trained from a lot of positive and negative images. 
It is then used to detect objects in other images. 
The concept of **Cascade Of Classifiers** is that Instead of applying all 6000 features on a window, the features are grouped into different stages of classifiers and applied one-by-one.  
If a window fails the first stage, discard it. 
We don't consider the remaining features on it. 
If it passes, apply the second stage of features and continue the process. 
The window which passes all stages is a face region.
In an image, most of the image is non-face region. 
So it is a better idea to have a simple method to check if a window is not a face region. 
If it is not, discard it in a single shot, and don't process it again. Instead, focus on regions where there can be a face.
## Run Locally

Clone the project

```bash
  git clone https://github.com/Sam8239/Covid-19_Mask_Detector.git
```

Go to the project directory

```bash
  cd Covid-19_Mask_Detector
```

Install dependencies

```bash
  pip install cvlib 
  pip install opencv-python
  pip install tensorflow
```

Start the project

```bash
  python .\mask_detector.py
```


  
## Screenshots
### Wearing a Mask
<img src="./Screenshot/wearing_mask" alt="logo" height="400px" width="400px"><br>

### Not Wearing a Mask
<img src="./Screenshot/not_wearing_mask" alt="logo" height="400px" width="400px"><br>

### No Face Detected
<img src="./Screenshot/no_face_detected" alt="logo" height="400px" width="400px">