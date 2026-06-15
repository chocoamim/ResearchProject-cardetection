"Michelle A. Oparaugo______5/20/2026"
"Car detection using YOLOv8+DeepSORT"

#important imports for the code
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import csv
import os
import math
import pandas as pd


#shows exact folder where output video is saved
print("Current working directory:", os.getcwd())

#this loads yolo v8 model
model = YOLO("yolov8n.pt")  # small + fast

#this creates and opens a csv file where the csv output will be saved in
csv_file = open("detections.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "track_id", "x_center", "y_center", "x1", "y1", "w", "h"])

#initialize deepsort
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
frame_log = {}

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

        if conf < 0.2:
            continue

        # Only track vehicles
        if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            w_box = x2 - x1
            h_box = y2 - y1
            x_center = int(x1 + w_box / 2)
            y_center = int(y1 + h_box / 2)

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)


    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()

        w_box = x2 - x1
        h_box = y2 - y1
        x_center = int(x1 + w_box / 2)
        y_center = int(y1 + h_box / 2)

        if frame_number not in frame_log:
            frame_log[frame_number] = {}
        frame_log[frame_number][track_id] = (x_center, y_center)

        fps = 25
        meters_per_pixel = 0.05

        # draws the trajectories
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((x_center, y_center))

        if len(trajectories[track_id]) >= 2:
            px1, py1 = trajectories[track_id][-2]
            px2, py2 = trajectories[track_id][-1]

            dist_pixels = math.sqrt((px2 - px1)**2 + (py2 - py1)**2)
            dist_meters = dist_pixels * meters_per_pixel
            speed_mps = dist_meters * fps
            speed_mph = speed_mps * 2.237
        else:
            speed_mph = 0

        #write to csv
        csv_writer.writerow([frame_number, track_id, x_center, y_center, int(x1), int(y1), int(w_box), int(h_box)])

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"{speed_mph: .1f} mph", (int(x1), int(y2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

'''CALCULATION SECTION'''

meters_per_pixel = 0.05
fps = 25

vehicle_speeds = {}
instant_speeds = {}

for track_id, points in trajectories.items():
    if len(points) < 2:
        continue

    instant_speeds[track_id] = []
    total_distance_pixels = 0

    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        dist_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance_pixels += dist_pixels

        dist_meters = dist_pixels * meters_per_pixel
        speed_mps = dist_meters * fps
        speed_mph = speed_mps * 2.237
        instant_speeds[track_id].append(speed_mph)

    total_time = len(points) / fps
    total_distance_meters = total_distance_pixels * meters_per_pixel
    avg_speed_mps = total_distance_meters / total_time
    avg_speed_mph = avg_speed_mps * 2.237

    vehicle_speeds[track_id] = avg_speed_mph

max_instant_speed = 0
max_instant_track = None

for track_id, speeds in instant_speeds.items():
    if len(speeds) == 0:
        continue
    local_max = max(speeds)
    if local_max > max_instant_speed:
        max_instant_speed = local_max
        max_instant_track = track_id

print('+++HIGHEST SPEED+++')
print(f"\nHighest instantaneous speed: {max_instant_speed:.2f} mph (Track {max_instant_track})")

print("\n****Average speed per vehicle(mph)****")
for tid, spd in vehicle_speeds.items():
    print(f'Track{tid}: {spd:.2f} mph')

print("\n****Instantaneous speeds(mph)****\n")
for tid, speeds in instant_speeds.items():
    print(f'Track {tid}: {['%.1f' % s for s in speeds]}')

cv2.destroyAllWindows()

'''GROUPING CSV OUTPUT SECTION'''

df = pd.read_csv("detections.csv")
df = df.sort_values(["track_id", "frame"])

with open("trajectories_by_id.csv", "w", newline="") as f:
    for tid, group in df.groupby("track_id"):
        f.write(f"Track ID: {tid}\n")
        group.to_csv(f, index=False)
        f.write("\n")

print("\nGrouped CSV saved as trajectories_by_id.csv")

#confirms output files
if os.path.exists("output_tracking.avi"):
    print("Output file size:", os.path.getsize("output_tracking.avi"), "bytes")
else:
    print("ERROR: output_tracking.avi was not created.")