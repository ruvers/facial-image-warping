import cv2

from backend.face_parsing import parse_face, get_mask

img = cv2.imread("test.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

parsing = parse_face(img)

hair_mask = get_mask(parsing, [17])

cv2.imwrite("hair_mask.png", hair_mask)

print("done")