import cv2

cap = cv2.VideoCapture('video.avi')
car_cascade = cv2.CascadeClassifier('cars.xml')
bus_cascade = cv2.CascadeClassifier('Bus_front.xml')
bike_cascade = cv2.CascadeClassifier('two_wheeler.xml')

print("Honda Hackathon")
print(" Video opening")
while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)
    bus = car_cascade.detectMultiScale(gray, 1.16, 1)
    bike = car_cascade.detectMultiScale(gray,1.01, 1)
    # if str(np.array(cars).shape[0]) == '1':
    #     i += 1
    #     continue
    for (x,y,w,h) in cars:
        plate = frames[y:y + h, x:x + w]
        cv2.rectangle(frames,(x,y),(x +w, y +h) ,(51 ,51,255),2)
        cv2.rectangle(frames, (x, y - 40), (x + w, y), (51,51,255), -2)
        cv2.putText(frames, 'Vehicle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('car',plate)    

    for (x,y,w,h) in bus:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
        cv2.putText(frames, 'Vehicle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('bus',plate)
        break

    for (x,y,w,h) in bike:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,215),2)
        cv2.putText(frames, 'Vehicle', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('bus',plate) 
    # lab1 = "Car Count: " + str(i)
    # cv2.putText(frames, lab1, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 3)
    frames = cv2.resize(frames,(600,400))
    cv2.imshow('Car Detection System', frames)
    # cv2.resizeWindow('Car Detection System', 600, 600)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
