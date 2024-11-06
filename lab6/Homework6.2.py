import cv2

def track_with_kcf(video_path, bbox, frame_count=20):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    tracker_kcf = cv2.TrackerKCF_create()
    tracker_kcf.init(frame, bbox)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox_kcf = tracker_kcf.update(frame)

        if success:
            p1 = (int(bbox_kcf[0]), int(bbox_kcf[1]))
            p2 = (int(bbox_kcf[0] + bbox_kcf[2]), int(bbox_kcf[1] + bbox_kcf[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        cv2.imshow('KCF Tracker', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_with_csrt(video_path, bbox, frame_count=20):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    tracker_csrt = cv2.TrackerCSRT_create()
    tracker_csrt.init(frame, bbox)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox_csrt = tracker_csrt.update(frame)

        if success:
            p1 = (int(bbox_csrt[0]), int(bbox_csrt[1]))
            p2 = (int(bbox_csrt[0] + bbox_csrt[2]), int(bbox_csrt[1] + bbox_csrt[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

        cv2.imshow('CSRT Tracker', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = 'C:\labsCV\lab6\\vid.mp4'

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return

    bbox = cv2.selectROI(frame, False)
    cv2.destroyAllWindows()

    track_with_kcf(video_path, bbox)
    track_with_csrt(video_path, bbox)


if __name__ == '__main__':
    main()
