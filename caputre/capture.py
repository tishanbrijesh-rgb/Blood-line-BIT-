"""
capture.py
Video input module for FaceTrap – handles webcam/video file capture.
"""

import cv2


class VideoCapture:
    def __init__(self, source=0, width=640, height=480):
        """
        Initialize video source.
        :param source: 0 = webcam, or path to video file
        :param width: frame width
        :param height: frame height
        """
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"[ERROR] Unable to open video source: {source}")

        # Set resolution (works for most webcams)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        """
        Reads a single frame from the video source.
        :return: frame if success, None if failed
        """
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def frame_generator(self):
        """
        Generator that continuously yields frames.
        Example:
            for frame in VideoCapture(0).frame_generator():
                ...
        """
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            yield frame

    def release(self):
        """Release the video capture object."""
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    # Test module
    cap = VideoCapture(0)

    for frame in cap.frame_generator():
        cv2.imshow("Video Input Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
