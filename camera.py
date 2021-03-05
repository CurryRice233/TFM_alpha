import socket
import threading
import cv2
import numpy
import struct

img_resolution = (600, 360)
img_fps = 30


class ClientThread(threading.Thread):
    def __init__(self, client_address, client_socket, camera):
        threading.Thread.__init__(self)
        self.client_address = client_address
        self.client_socket = client_socket
        self.camera = cv2.VideoCapture(camera)

    def run(self):
        try:
            while True:
                ret, img = self.camera.read()
                if ret:
                    img = cv2.resize(img, img_resolution)  # reset img resolution
                    cv2.imshow("Camera", img)
                    # _, img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 20])  # encode img
                    img = cv2.imencode('.jpg', img)[1]
                    img = numpy.array(img).tobytes()  # img to string

                    # send img with len and resolution
                    self.client_socket.send(struct.pack('lhh', len(img), img_resolution[0], img_resolution[1]) + img)
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
                else:
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # break
        except (ConnectionAbortedError, ConnectionResetError):
            print('{} closed connection'.format(self.client_address))
            pass
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            self.client_socket.close()


server_config = ('127.0.0.1', 7788)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(server_config)
server.listen(5)
print('Waiting for client')
while True:
    clientSocket, clientAddress = server.accept()
    newThread = ClientThread(clientAddress, clientSocket, "./test_video/1.MOV")
    newThread.start()
