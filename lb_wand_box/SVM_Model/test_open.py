import cv2
import numpy as np

image = 255 * np.ones((500, 500, 3), dtype=np.uint8)

cv2.imshow('test window', image)

cv2.waitKey(0)

cv2.destroyAllWindows()
