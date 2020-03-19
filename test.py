import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import pdb
from skimage import measure

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def find_edges(img):
    v = np.median(img)
    sigma = 0.33
    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(img, 100, 200)
    return edges
def find_connected_components(img):
    # pdb.set_trace()
    L = 0
    A = img
    width = img.shape[1]
    length = img.shape[0]
    Q = np.zeros(A.shape)
    EQ = np.array([])
    if A[0,0] > 0:
        L+=1
        EQ  = np.append(EQ, L)
        Q[0,0] = L
    for x in range(1, width):
        if A[0,x] > 0 and A[0,x] == A[0, x-1]:
            Q[0,x] = Q[0,x-1]
        if A[0,x] > 0 and A[0,x] != A[0, x-1]:
            L+=1
            EQ  = np.append(EQ, L)
            Q[0,x] = L
    for y in range(1, length):
        if A[y,0] > 0 and A[y,0] == A[y-1,0]:
            Q[y,0] = Q[y-1,0]
        if A[y,0] > 0 and A[y,0] != A[y-1,0]:
            L+=1
            EQ  = np.append(EQ, L)
            Q[y,0] = L
        for x in range(1,width):
            p = A[y,x]
            left = A[y,x-1]
            qleft = Q[y,x-1]
            up = A[y-1,x]
            qup = Q[y-1,x]
            lu = A[y-1,x-1]
            qlu = Q[y-1,x-1]
            if x == img.shape[1] - 2 or x == img.shape[1] - 1:
                ru = A[y-1,x]
                qru = Q[y-1,x]
            else:
                ru = A[y-1,x+1]
                qru = Q[y-1,x+1]
            if p > 0:
                if up == 0 and left == 0 and lu == 0 and ru == 0:
                    L+=1
                    EQ  = np.append(EQ, L)
                    Q[y,x] = L
                else :
                    if p == left and p == up and p == lu and p == ru and (qleft != qup or qup != qlu or qleft != qlu or qru!= qlu or qru != qup or qru!=qleft):
                        pixels = [qleft,qup,qlu, qru]
                        pixels.sort()
                        L1 = pixels[0]
                        L2 = pixels[1]
                        L3 = pixels[2]
                        L4 = pixels[3]
                        Q[y,x] = L1
                        EQ[int(L2)-1] = L1
                        EQ[int(L3)-1] = L1
                        EQ[int(L4)-1] = L1

                    if p == left and p == up and p == lu and p == ru and (qleft == qup and qup == qlu and qleft == qlu and qru== qlu and qru==qup and qru==qleft):
                        Q[y,x] = qleft

                    if (p == left and p == ru) and qleft != qru:
                        pixels = [qleft,qru]
                        pixels.sort()
                        L1 = pixels[0]
                        L2 = pixels[1]
                        Q[y,x] = L1
                        EQ[int(L2)-1] = L1

                    if p == left:
                        Q[y,x] = qleft
                    if p == up:
                        Q[y,x] = qup
                    if p == lu:
                        Q[y,x] = qlu
                    if p == ru:
                        Q[y,x] = qru

    for i in range(len(EQ),1,-1):
        if (i) != EQ[i-1]:
            Q[Q==(i)] = EQ[i-1]
    pdb.set_trace()
    return Q, EQ
def color_components(img, wait):
    # pdb.set_trace()
    eq = len(np.unique(img))
    nimg = np.expand_dims(img, axis = 2)
    nimg = np.insert(nimg, 0, 255, axis=2)
    nimg = np.insert(nimg, 0, 255, axis=2)
    # nimg = rainbow_circles
    # nimg[img==0] = [0,0,0]

    rainbow_list = [[0, 0 , 0 ],
        [148, 0, 211],
        [75, 0, 130],
        [0, 0, 255],
        [40, 255, 0],
        [55, 255, 0],
        [255, 127, 0],
        [255, 0 , 0 ]]
    nimg = np.array(rainbow_list)[img%len(rainbow_list)]
    return nimg

def capture_camera():
    # pdb.set_trace()
    # rainbow_circles = cv2.imread('./images/rainbow_circles.jpg')
    # plt.imshow(rainbow_circles)
    # plt.show()

    # Open the device at the ID 0
    cap = cv2.VideoCapture(0)

    #Check whether user selected camera is opened successfully.
    if not (cap.isOpened()):
        print("Could not open video device")

    #To set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        edges = find_edges(frame)
        connected_components = measure.label(edges)
        colored_image = color_components(connected_components, 10).astype(np.uint8)
        cv2.imshow('preview',colored_image)

        #Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_camera()

