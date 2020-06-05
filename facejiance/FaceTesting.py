import cv2




def faceRecognition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 把人连彩色图片转化成灰度图片
    # cv2.imshow("DengZiQi", gray) 显示灰度视频
    # OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
    color = (0, 255, 0)  # 定义框出人脸方框的颜色
    # 调用识别人脸，并返回所有人脸
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸 cv2.rectangle(img,p1,p2,color,thickness,linetype,shift)
            #该函数表示画出一个矩形，p1,p2分别表示图像左上角和右下角的点,color表示线的颜色，thickness表示线的粗度，shift 参数表示点坐标中的小数位数
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 3)
            # 左眼，画圆
            cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color,3)
            # 右眼
            cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color,3)
            # 嘴巴
            cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color,3)
    cv2.imshow("image", img)  # 显示图像


# 如果传入为0则表示调用计算机内置摄像头，如果传入1，则表示调用外置摄像头，比如usb连接的摄像头
cap = cv2.VideoCapture(0)
while 1:  # 循环显示摄像头数据，逐帧显示
    ret, img = cap.read()  #ret(第一个返回值)表示是否获得图像数据 img表示获得的一帧的图像
    # cv2.imshow("Image", img)
    faceRecognition(img)  #对每一帧图像进行处理
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源
