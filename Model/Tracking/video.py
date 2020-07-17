import cv2

class Video:
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = -1

        self.w, self.h = 1920, 1080
        print("Length of the video: {:d} frames".format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def get_one_frame(self):
        _, frame = self.cap.read()
        self.count += 1
        return frame

    def __iter__(self):
        # self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, frame = self.cap.read()  # BGR
        assert frame is not None, 'Failed to load frame {:d}'.format(self.count)
        # img0 = frame[:, :, ::-1].transpose(2, 0, 1)

        return frame

    def __len__(self):
        return self.vn
