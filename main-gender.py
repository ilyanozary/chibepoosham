# import pac
import cv2
from deepface import DeepFace

img = cv2.imread("iamges-test/123.jpg")

result = DeepFace.analyze(img, actions=("gender", 'age'))

print(result)
