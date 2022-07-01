from PIL import ImageGrab, Image
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import pyperclip as pc

ocr = PaddleOCR(lang='japan')

while True:
    im = ImageGrab.grabclipboard()
    img_path = 'tem.png'
    im.save(img_path)
    result = ocr.ocr(img_path)

    rs = "\n".join([e[1][0] for e in result])

    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
    image = np.array(im_show)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("", image)
    print(rs)
    pc.copy(rs)
    cv2.waitKey(0)
