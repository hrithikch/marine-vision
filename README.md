Using ultralytics yolov8 with modd2 data for bounding boxes

yolo track model=yolov8s.pt source=data\raw\seg_vid\crop11.mp4 tracker=botsort.yaml save=True save_txt=True conf=0.1 project=runs\track name=boats

save_txt saves box per frame.
