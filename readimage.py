import cv2

image = cv2.imread("./out.png", cv2.IMREAD_ANYCOLOR)

print(image.shape)

cv2.imshow("Moon", image)
cv2.waitKey()
cv2.destroyAllWindows()