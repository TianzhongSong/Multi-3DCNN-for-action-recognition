import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
from ssd_utils import BBoxUtility
from ssd import SSD300 as SSD
import os

def run_camera(input_shape,model,video_path, image_path_ori, image_path_crop):
    num_classes = 21
    conf_thresh = 0.5
    input_shape = input_shape
    bbox_util = BBoxUtility(num_classes)

    class_colors = []
    for i in range(0,num_classes):
        hue = 255 * i / num_classes
        col = np.zeros((1, 1, 3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128  # Saturation
        col[0][0][2] = 255  # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)

    class_list = os.listdir(video_path)
    for action in class_list:
        all_action = os.listdir(video_path+action)
        for sample in all_action:
            print(video_path+action+'/'+sample)
            name = sample.split('.')[0]
            if not os.path.exists(image_path_ori+action+'/'+name):
                os.mkdir(image_path_ori+action+'/'+name)
            if not os.path.exists(image_path_crop+action+'/'+name):
                os.mkdir(image_path_crop+action+'/'+name)
            vid = cv2.VideoCapture(video_path+action+'/'+sample)

            # Compute aspect ratio of video
            vidw = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
            vidh = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
            frame_length = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            # vidar = vidw / vidh
            frame_count = 0
            for n in range(int(frame_length)):
                retval, orig_image = vid.read()
                if not retval:
                    print("Done!")
                    return

                im_size = (input_shape[0], input_shape[1])
                resized = cv2.resize(orig_image, im_size)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


                inputs = [image.img_to_array(rgb)]
                tmp_inp = np.array(inputs)
                x = preprocess_input(tmp_inp)

                y = model.predict(x)


                results = bbox_util.detection_out(y)

                if len(results) > 0 and len(results[0]) > 0:
                    det_label = results[0][:, 0]
                    det_conf = results[0][:, 1]
                    det_xmin = results[0][:, 2]
                    det_ymin = results[0][:, 3]
                    det_xmax = results[0][:, 4]
                    det_ymax = results[0][:, 5]

                    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

                    top_conf = det_conf[top_indices]
                    top_label_indices = det_label[top_indices].tolist()
                    top_xmin = det_xmin[top_indices]
                    top_ymin = det_ymin[top_indices]
                    top_xmax = det_xmax[top_indices]
                    top_ymax = det_ymax[top_indices]

                    if 15 not in top_label_indices:
                        pass
                    else:
                        for i in range(top_conf.shape[0]):
                            xmin = int(round((top_xmin[i] * vidw) * 0.9))
                            ymin = int(round((top_ymin[i] * vidh) * 0.9))
                            xmax = int(round((top_xmax[i] * vidw) * 1.1)) if int(
                                round((top_xmax[i] * vidw) * 1.1)) <= vidw else int(
                                round(top_xmax[i] * vidw))
                            ymax = int(round((top_ymax[i] * vidh) * 1.1)) if int(
                                round((top_ymax[i] * vidh) * 1.1)) <= vidh else int(
                                round(top_ymax[i] * vidh))

                            # save frames
                            class_num = int(top_label_indices[i])
                            if class_num == 15:
                                frame = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
                                cv2.imwrite(image_path_ori+action+'/'+name+str(10000+frame_count)+'.jpg',frame)
                                cropImage = frame[ymin:ymax, xmin:xmax]
                                cropImage = cv2.resize(cropImage, (64, 64))
                                cv2.imwrite(image_path_crop + action + '/' + name + str(10000+frame_count) + '.jpg', cropImage)
                                frame_count += 1


if __name__ == '__main__':
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                   "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    video_path = '/home/deep/datasets/kth/video/'
    save_path_ori = '/home/deep/datasets/kth/images/original/'
    save_path_crop = '/home/deep/datasets/kth/images/crop/'
    input_shape = (300, 300, 3)
    NUM_CLASSES = len(class_names)
    ssd_model = SSD(input_shape, num_classes=NUM_CLASSES)
    ssd_model.load_weights('weights_SSD300.hdf5')
    run_camera(input_shape, ssd_model, video_path, save_path_ori, save_path_crop)