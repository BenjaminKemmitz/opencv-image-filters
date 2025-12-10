import cv2

def orb_features(image_path, n_features=500):
    img = cv2.imread(image_path)
    orb = cv2.ORB_create(nfeatures=n_features)

    keypoints, descriptors = orb.detectAndCompute(img, None)
    out = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))

    return out

if __name__ == "__main__":
    img = orb_features("../images/input/sample.jpg")
    cv2.imwrite("../images/output/orb_features.jpg", img)

