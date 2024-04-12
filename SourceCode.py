import cv2

sift = cv2.SIFT_create()

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Load images
img1 = cv2.imread('test1.jpg')
img2 = cv2.imread('test.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3_resized = cv2.resize(img3, (1200, 600))
cv2.imwrite('sift_output.jpg', img3_resized)

# Display result
cv2.imshow('SIFT', img3_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

