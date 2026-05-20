"Michelle A. Oparaugo______5/20/2026"
"Car detection using YOLOv8+DeepSORT"

#important imports for the code
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import csv
import os

#shows exact folder where output video is saved
print("Current working directory:", os.getcwd())

#this loads yolo v8 model
model = YOLO("yolov8n.pt")  # small + fast

#this creates and opens a csv file where the csv output will be saved in
csv_file = open("detections.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

#initialixe deepsort
tracker = DeepSort(max_age=30)

#opens the input video
cap = cv2.VideoCapture("2.avi")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#creates a video writer for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_tracking.avi", fourcc, 20.0, (w, h))

#main loop
frame_number = 0

trajectories = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]

        csv_writer.writerow([frame_number, cls, model.names[cls], conf,
                             float(x1), float(y1), float(x2), float(y2)])

        # Only track vehicles
        if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)


    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        #draws the trajectories
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))


        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    dead_tracks = [tid for tid in list(trajectories.keys())
                   if not any(t.track_id == tid and t.is_confirmed() for t in tracks)]
    for tid in dead_tracks:
        del trajectories[tid]

    for track_id, points in trajectories.items():
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)

    out.write(frame)

    cv2.putText(frame, f"Frame: {frame_number}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 0, 255), 2)

    cv2.imshow("YOLO + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

#confirms output files
if os.path.exists("output_tracking.avi"):
    print("Output file size:", os.path.getsize("output_tracking.avi"), "bytes")
else:
    print("ERROR: output_tracking.avi was not created.")
