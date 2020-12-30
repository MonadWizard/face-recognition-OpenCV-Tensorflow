
# Real-time face Recognition Using Facenet On Tensorflow 2.X

This is a quick guide of how to get set up and running a robust real-time facial recognition system using the Pretraiend Facenet Model and MTCNN.

0. Make a directory of your name inside the Faces folder and upload your 2-3 pictures of you.
or,
1. run ```create_train_image.py``` for create image data-set to train
2. Run ``` train_v2.py```.
3. Then run ```detect.py``` for realtime face recognization.

As the Facenet model was trained on older versions of TensorFlow, the architecture.py file is used to define the model's architecture on newer versions of TensorFlow so that the pre-trained model's weight can be loaded.
<br>

Dowload pre-trained weight from [Here.👈](https://drive.google.com/drive/folders/1-Frhel960FIv9jyEWd_lwY5bVYipizIT) <br>

For in depth explanation follow this amazingly expained [article. 👈](https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/)

# Dependencies
This code was working properly on tensroflow 2.3.0.
```
tenorflow 2.X
numpy
opencv-python
mtcnn
scikit-learn
scipy
```
### Credit: https://github.com/Practical-AI/Face



