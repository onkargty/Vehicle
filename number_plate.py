import cv2
import easyocr

# Haar cascade for number plate detection
harcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height

min_area = 500
count = 0

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

while True:
    success, img = cap.read()
    if not success:
        break

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save the captured image of the number plate
        img_path = "plates/scanned_img_" + str(count) + ".jpg"
        cv2.imwrite(img_path, img_roi)
        
        # Perform OCR on the saved image using EasyOCR
        ocr_result = reader.readtext(img_roi)
        ocr_text = ' '.join([res[1] for res in ocr_result])
        print("OCR Result: ", ocr_text)
        
        # Display the OCR result on the image
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved: " + ocr_text, (50, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1

cap.release()
cv2.destroyAllWindows()