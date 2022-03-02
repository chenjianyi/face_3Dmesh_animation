# encoding:utf-8
import requests 
import base64
import cv2

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=YnV45N8oVS9VzP7djULGqiK0&client_secret=IPAxiVzvIgiYEjUIUcd0uO6xrufPDyRF'
response = requests.get(host)
if response:
    print(response.json())

img = cv2.imread('')

data = {
    "version": 1,
    "log_id": 234,
    "images": [
                "base64(img1)",
          "base64(img2)"
    ],
    "depths": [
                "base64(depth1)",
          "base64(depth2)"
    ],
          "params": [
                "base64(param1)",
          "base64(param2)"
    ]
}

r = requests.post(host, data=data)
print(r.text)
