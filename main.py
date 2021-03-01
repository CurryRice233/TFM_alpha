import socket
import struct
import numpy
import cv2
import threading
import warnings
from detection.detection_manager import DetectionManager

warnings.simplefilter(action='ignore', category=FutureWarning)
DM = DetectionManager()

camera_ip = ('127.0.0.1', 7788)
camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
camera_socket.connect(camera_ip)


class DetectionThread(threading.Thread):
    def __init__(self, algorithm_id, title):
        threading.Thread.__init__(self)
        self.isDetecting = False
        self.algorithm_id = algorithm_id
        self.title = title
        self.stop = False

    def run(self):
        while True:
            try:
                if self.stop:
                    break
                if 'img' in globals() and 'new_img' in globals():
                    global img, new_img
                    if new_img:
                        self.isDetecting = True
                        cv2.imshow(str(self.title), DM.run_detect(self.algorithm_id, img))
                        self.isDetecting = False
                        new_img = False
            finally:
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


try:
    detection_thread = DetectionThread(0, camera_ip)
    detection_thread.start()
    while True:
        img_info = struct.unpack("lhh", camera_socket.recv(8))
        buf_size = img_info[0]  # get size of img
        buf = b''
        while buf_size:
            temp_buf = camera_socket.recv(buf_size)
            buf_size -= len(temp_buf)
            buf += temp_buf
            if not detection_thread.isDetecting:
                global img, new_img
                img = numpy.frombuffer(buf, dtype='uint8')
                img = cv2.imdecode(img, 1)
                new_img = True
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
except (struct.error, ConnectionAbortedError, ConnectionResetError):
    detection_thread.stop = True
    detection_thread.join()
    pass
finally:
    camera_socket.close()
    cv2.destroyAllWindows()