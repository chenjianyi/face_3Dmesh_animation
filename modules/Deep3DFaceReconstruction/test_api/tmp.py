import os
import cv2

tex = cv2.imread('tex.jpg')
h, w, c = tex.shape
print(tex.shape)

color_list = []
with open('face.obj', 'r') as f:
    for line in f:
        if line.startswith('vt '):
            tmp = line.strip().split()
            
            x, y = int(w - float(tmp[1]) * w), int(h - float(tmp[2]) * h)
            x = min(max(x, 0), w-1)
            y = min(max(y, 0), h-1)
            b, g, r = tex[y][x]
            #print(tmp, x, y, b, g, r)
            color_list.append([r, g, b])

new_f = open('new_face.obj', 'w')
with open('face.obj', 'r') as f:
    i = 0
    for line in f:
        if line.startswith('v '):
            color = color_list[i]
            b, g, r = color
            new_line = '%s %s %s %s' % (line.strip(), b, g, r)
            print(new_line, file=new_f)
            i = i + 1
        elif line.startswith('vt '):
            continue
        else:
            new_line = line.strip()
            print(new_line, file=new_f)
new_f.close()
