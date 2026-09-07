import cv2
import numpy as np
import re

def preprocess_pan_image(image, debug=False):

    # 1. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if debug:
        cv2.imshow("1. Grayscale", gray)

    # 2. Bilateral Filter (preserves edges, reduces noise)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    if debug:
        cv2.imshow("2. Bilateral Filter", filtered)

    # 3. Contrast Enhancement (TopHat + BlackHat)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel, iterations=2)
    blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel, iterations=2)
    contrast_enhanced = cv2.add(filtered, tophat)
    contrast_enhanced = cv2.subtract(contrast_enhanced, blackhat)
    if debug:
        cv2.imshow("3. Contrast Enhanced", contrast_enhanced)

    # 4. Adaptive Thresholding (clean binarization for text)
    binary = cv2.adaptiveThreshold(
        contrast_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    if debug:
        cv2.imshow("4. Adaptive Threshold", binary)

    # 5. Morphological Closing to reconnect broken characters
    morph_kernel = np.ones((1, 1), np.uint8)  # Tuned kernel size
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)
    if debug:
        cv2.imshow("5. Morphological Closing", closed)

    # 6. Final Inversion for CRNN (black text on white bg)
    inverted = cv2.bitwise_not(closed)
    if debug:
        cv2.imshow("6. Inverted Output", inverted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return inverted

def is_bad_text(text: str, expected_format: str = "") -> bool:

    text = text.strip()

    # Completely blank
    if not text:
        return True

    # PAN format check
    if expected_format == "pan":
        return not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text)

    # DOB format check
    elif expected_format == "dob":
        return not re.fullmatch(r"\d{2}/\d{2}/\d{4}", text)

    # Name check: very short or digits-only (garbage)
    elif expected_format == "name":
        if len(text) < 3 or text.isdigit():
            return True

    # Fallback: text too short or unreadable
    return len(text) < 3

def skew_correction(image, debug=False):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle_deg = (theta * 180 / np.pi) - 90
            if -45 < angle_deg < 45:
                angles.append(angle_deg)
        if angles:
            angle = np.median(angles)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return deskewed
