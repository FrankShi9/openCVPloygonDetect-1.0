import cv2
import math

threshhold_value = 60
coefficient = 0.02

class ShapeDetector:

    def __init__(self):
        #字典counter对应每一种图形的计数器
        self.counter = {"unrecognized image": 0, "triangle": 0, "rhombus": 0,"rectangle": 0, "pentagon": 0, "hexagon": 0,"circle": 0}
        #初始化shape为不可识别
        self.shape = "unrecognized image"
        #图形顶点集置空
        self.vertices = []
        #初始化该图形的周长为0
        self.peri = 0

    #计算欧式距离(主要作用通过计算边长区分菱形和长方形)
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def detect(self, c):
        #cv2.arcLength函数返回周长
        self.peri = cv2.arclength(c, True)
        #cv2.approxPolyDP用多边形取拟合，返回的是顶点的列表
        self.vertices = cv2.approxPolyDP(c, coefficient*self.peri, True)


        if len(self.vertices) == 3:
            self.shape = "triangle"


        elif len(self.vertices) == 4:

            dist1 = self.distance(self.vertices[0][0][0], self.vertices[0][0][1], self.vertices[1][0][0],self.vertices[1][0][1])

            dist2 = self.distance(self.vertices[0][0][0], self.vertices[0][0][1], self.vertices[3][0][0],self.vertices[3][0][1])

            result = math.fabs(dist1 - dist2)

            if result <= 10:
                self.shape = "rhombus"
            else:
                self.shape = "rectangle"

        elif len(self.vertices) == 5:
            self.shape = "pentagon"

        elif len(self.vertices) == 6:
            self.shape = "hexagon"

        else:
            self.shape = "circle"

        self.counter[self.shape] += 1

        return self.shape


    def display(self):

        for type in self.counter.keys():
            print("The number of {} is {}".format(type, self.counter[type]))

    def main():
        #读入图片
        testData = "polygon.png"
        image = cv2.imread(testData)

        #将图片转换为灰度图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #高斯滤波,图像平滑处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #根据阈值，将灰度图片转化为黑白两色图片
        thresh = cv2.threshold(blurred, threshhold_value, 255, cv2.THRESH_BINARY_INV)[1]

        #返回图片和图中轮廓信息，列表形式返回到cnts中
        counts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         #只需要取轮廓上点的信息

        counts = counts[1]

        #创建一个识别器实例
        test_sd = ShapeDetector()

        #分别对每个轮廓进行处理
        for c in counts:

            shape = test_sd.detect(c)

        test_sd.display()