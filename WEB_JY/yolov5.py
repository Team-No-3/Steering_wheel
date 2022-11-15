import torch, cv2, pandas

model = torch.hub.load("ultralytics/yolov5", "custom", path="flask_deep/models/yolov5/yolov5.pt")

def plot_boxes(predicts, frame):
    for row in predicts:
        # Confidence
        if row[4] >= 0.1:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            rgb = (0, 255, 0)
            # classid
            if row[5] == 1:
                rgb = (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), rgb, 2)
            cv2.putText(frame, row[6], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rgb, 2)
    return frame