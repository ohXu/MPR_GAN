# coding=gbk
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from PIL import Image
import cv2
import numpy
import time
import math

EYE = numpy.array([0.0, 0.06, -0.08])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = numpy.array([5.0, 0.06, -0.55])  # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = numpy.array([0.0, 0.0, 1.0])
MOUSE_X, MOUSE_Y = 0, 0
time_last = time.time()


class point_cloud:

    def __init__(this, file_path):
        this.file_path = file_path
        this.bCreate = False
        this.GAN = True

    def createVAO(this):

        indexes = []
        p = numpy.loadtxt(this.file_path, dtype=numpy.float32)
        for i in range(p.shape[0]):
            indexes.append(i)
        # p1 = p.reshape(p.shape[0] * p.shape[1])
        this.vbo = vbo.VBO(p)
        this.ebo = vbo.VBO(numpy.array(indexes, numpy.uint32), target=GL_ELEMENT_ARRAY_BUFFER)
        this.eboLength = len(indexes)
        this.bCreate = True

    def draw(this):

        if this.bCreate == False:
            this.createVAO()
        this.vbo.bind()
        glInterleavedArrays(GL_C3F_V3F, 0, None)
        this.ebo.bind()
        glDrawElements(GL_POINTS, this.eboLength, GL_UNSIGNED_INT, None)


def drawFunc2():
    global pc, time_last, pc_number

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(20.0, 1224.0 / 256.0, 1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glColor3f(1.0, 0.0, 0.0)
    gluLookAt(
        EYE[0], EYE[1], EYE[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )
    glPointSize(8)
    pc.draw()
    glutSwapBuffers()

    # h = 256
    # w = 1024
    # data = glReadPixels(0, 0, 1024, 256, GL_RGB, GL_UNSIGNED_BYTE)
    # arr = numpy.frombuffer(data, dtype=numpy.uint8)
    # arr = numpy.reshape(arr, (h, w, 3))
    # cv2.flip(arr, 0, arr)
    # arr = (arr / 127.5) - 1
    # img = tf.convert_to_tensor(arr, dtype=tf.float32)
    # img = tf.expand_dims(img, 0)
    # ig = generate_image(img, generator)
    # ig = cv2.cvtColor(numpy.asarray(ig), cv2.COLOR_RGB2BGR)
    # cv2.imshow('scene', ig)
    # cv2.moveWindow("scene", 0, 300)
    # cv2.waitKey(1)


def open_image(file_path):
    img = Image.open(file_path)
    img.show()
    return numpy.array(img)


def getposture():
    global EYE, LOOK_AT
    dist = numpy.sqrt(numpy.power((EYE - LOOK_AT), 2).sum())

    if dist > 0:

        phi = numpy.arcsin((LOOK_AT[2] - EYE[2]) / dist)
        theta = numpy.arccos((LOOK_AT[0] - EYE[0]) / (dist * numpy.cos(phi)))

    else:
        phi = 0.0
        theta = 0.0

    # print(phi * 180.0 / math.pi, theta * 180.0 / math.pi)
    return dist, phi, theta


def mouse(button, state, x, y):
    global MOUSE_X, MOUSE_Y
    MOUSE_X, MOUSE_Y = x, y

    if state == 0:
        pc.GAN = True
    else:
        pc.GAN = False
        # print(LOOK_AT)


def mouse_callback(x, y):

    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA

    dx = x - MOUSE_X
    dy = y - MOUSE_Y
    MOUSE_X, MOUSE_Y = x, y

    PHI += dy * 0.001
    THETA += dx * 0.001
    # THETA += 2 * numpy.pi * dx
    # THETA %= 2 * numpy.pi
    # r = DIST * numpy.cos(PHI)
    PHI = min(PHI, math.pi / 2.0)
    PHI = max(PHI, -math.pi / 2.0)
    # THETA = min(THETA, math.pi / 2.0)
    # THETA = max(THETA, -math.pi / 2.0)

    # print(PHI * 180.0 / math.pi, THETA * 180.0 / math.pi)
    LOOK_AT[0] = EYE[0] + DIST * math.cos(THETA) * math.cos(PHI)
    LOOK_AT[1] = EYE[1] + DIST * math.sin(THETA) * math.cos(PHI)
    LOOK_AT[2] = EYE[2] + DIST * math.sin(PHI)
    # print(LOOK_AT)

    glutPostRedisplay()


def generate_image(input_imgge, model):
    start = time.time()
    prediction = model(input_imgge, training=True)
    print('Time taken for GAN is {} sec\n'.format(time.time() - start))
    img2 = (prediction[0].numpy() * 0.5 + 0.5) * 255.0
    return prediction[0].numpy() * 0.5 + 0.5
    # img2 = Image.fromarray(img2.astype('uint8'))
    # img2.show()


def keydown(key, x, y):
    if key == b'z':
        h = 256
        w = 1024
        data = glReadPixels(0, 0, 1024, 256, GL_RGB, GL_UNSIGNED_BYTE)
        arr = numpy.zeros((h * w * 3), dtype=numpy.float32)
        for i in range(0, len(data), 3):
            arr[i] = data[i + 2]
            arr[i + 1] = data[i + 1]
            arr[i + 2] = data[i]

        arr = numpy.reshape(arr, (h, w, 3))
        cv2.flip(arr, 0, arr)
        cv2.imwrite('./1.png', arr)

    # if key == b'x':
    #     cv2.imwrite('C:/Users/lEGION/Desktop/Data/street_depth_img/' + str(pc_number).zfill(6) + '.png', arr)


pc = point_cloud('./demo.txt')
DIST, PHI, THETA = getposture()  # 眼睛与观察目标之间的距离、仰角、方位角

pc_number = 0

if __name__ == '__main__':

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutInitWindowPosition(0, 0)
    glutInitWindowSize(1024, 256)
    mainWindow = glutCreateWindow(b'XXX')
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glutMouseFunc(mouse)
    glutMotionFunc(mouse_callback)
    glutKeyboardFunc(keydown)
    # glutMouseWheelFunc(on_wheel)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glutDisplayFunc(drawFunc2)
    glutIdleFunc(drawFunc2)
    glutMainLoop()
