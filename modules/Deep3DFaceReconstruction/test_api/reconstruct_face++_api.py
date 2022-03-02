import requests
import json
import base64

url = "https://api-cn.faceplusplus.com/facepp/v1/3dface"

data = {
    "api_key": "kU2UD2bo0Js2yG3kT0mI8Pgo4LYZFeBs",
    "api_secret": "qq7eaiYi3qiwvK5zjUfaIRTcNAnW891W",
    "texture": 1,
    "mtl": 1
}

files = {
    "image_file_1": open("../WechatIMG19.jpeg", "rb"),
    "texture": 1,
    "mtl": 1
}

res = requests.post(url, data=data, files=files)

res_data = res.json()

print(res_data.keys())
print(res_data["transfer_matrix"])
print(res.status_code)

obj = str(base64.b64decode(res_data['obj_file']), "utf-8")
obj_file = open('face.obj', 'w')
print(obj, file=obj_file)
obj_file.close()

mtl_file = open('face.mtl', 'w')
mtl = str(base64.b64decode(res_data['mtl_file']), "utf-8")
print(mtl, mtl_file)
mtl_file.close()

tex = base64.b64decode(res_data['texture_img'])
#tex = res_data['texture_img']
tex_file = open('tex.jpg', 'wb')
tex_file.write(tex)
tex_file.close()
