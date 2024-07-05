import math
import os
from io import BytesIO
from uuid import uuid4

import numpy as np
import requests
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from paddleocr import PaddleOCR

from label_studio_ml.model import LabelStudioMLBase

global_ocr_instance = PaddleOCR(ocr_version='PP-OCRv4', use_gpu=True, use_multiprocess=False)


def get_exif_data(image):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value

    return exif_data


def load_image_from_url(url, token):
    headers = {'Authorization': f'Token {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        # 获取EXIF数据
        exif_data = get_exif_data(image)

        if 'Orientation' in exif_data:
            orientation = exif_data['Orientation']
            # 将图片旋转到正常方向
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

        return image
    else:
        raise Exception(f"Error loading image from {url}")


def convert_points_to_relative_xywhr(points, image):
    """
    Convert a list of points representing a rectangle to relative x, y, width, height, and rotation.
    The values are relative to the dimensions of the given image.
    Points are expected to be in the order: top-left, top-right, bottom-right, bottom-left.
    The rotation is calculated as the clockwise angle between the top edge and the horizontal line.
    Args:
    - points (list of lists): A list of four points, each point is a list of two coordinates [x, y].
    - image (numpy array): An image array.
    Returns:
    - tuple: (x, y, width, height, rotation) where x and y are the relative coordinates of the top-left point,
      width and height are the relative dimensions of the rectangle, and rotation is the angle in degrees.
    """
    # Extracting points
    top_left, top_right, bottom_right, bottom_left = points

    # Image dimensions
    img_height, img_width = image.shape[:2]

    # Calculate width and height of the rectangle
    width = math.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2)
    height = math.sqrt((bottom_right[0] - top_right[0]) ** 2 + (bottom_right[1] - top_right[1]) ** 2)

    # Calculate rotation in radians
    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]
    angle_radians = math.atan2(dy, dx)

    # Convert rotation to degrees
    rotation = math.degrees(angle_radians)

    # The top-left point is the origin (x, y)
    x, y = top_left

    # Convert dimensions to relative values (percentage of image dimensions)
    rel_x = x / img_width * 100
    rel_y = y / img_height * 100
    rel_width = width / img_width * 100
    rel_height = height / img_height * 100

    return rel_x, rel_y, rel_width, rel_height, rotation


class NewModel(LabelStudioMLBase):
    def __init__(self, project_id=None, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        self.ocr = global_ocr_instance
        self.token = os.environ['LABEL_STUDIO_API_KEY']
        self.label_studio_url = os.environ['LABEL_STUDIO_URL']

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", 'PP-OCRv4')

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            # '''坑点2：读取图像文件。虽然label studio和模型在同一台服务器上，但是在不同的端口。这样就导致了：（1）label studio上传图片时，无法直接加载模型服务器目录下的图片；
            # (2)模型后端无法直接读取label studio中上传的图片，source中显示的直接上传的图片目录为"/data/upload/12/bf68a25f-0034.jpg"。因此这里选择通过request
            # 请求获取数据。这里还有个小坑，每个账号有不同的token，请求的时候需要带上'''
            image_path = task['data']['image']
            image_url = self.label_studio_url + image_path
            image = load_image_from_url(image_url, self.token)
            # 使用OCR模型处理图像
            ocr_results = self.ocr.ocr(np.array(image))

            # 转换OCR结果为Label Studio所需的格式
            predictions = []
            # '''坑点3，必须带上id，上面说了，ocr任务有三个结果，如果没有id，前端就变成了3个结果'''
            for result in ocr_results[0]:
                ocr_id = str(uuid4())
                points, text_score = result
                text, score = text_score

                x, y, width, height, rotation = convert_points_to_relative_xywhr(points, np.array(image))

                # 标签（Labels）组件预测
                label_prediction = {
                    'from_name': 'label',
                    'id': str(ocr_id),
                    'to_name': 'image',
                    'type': 'labels',
                    'value': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'rotation': rotation,
                        'labels': ['Text']
                    }
                }

                # 矩形框（Rectangle）组件预测

                rectangle_prediction = {
                    'from_name': 'bbox',
                    'id': str(ocr_id),
                    'to_name': 'image',
                    'type': 'rectangle',
                    'value': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'rotation': rotation
                    }
                }

                # 文本区域（TextArea）组件预测
                textarea_prediction = {
                    'from_name': 'transcription',
                    'id': str(ocr_id),
                    'to_name': 'image',
                    'type': 'textarea',
                    'value': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'rotation': rotation,
                        'text': [text]
                    }
                }

                predictions.extend([label_prediction, rectangle_prediction, textarea_prediction])

            results.append({
                'result': predictions
            })

        return results

    # def fit(self, event, data, **kwargs):
    #     """
    #     This method is called each time an annotation is created or updated
    #     You can run your logic here to update the model and persist it to the cache
    #     It is not recommended to perform long-running operations here, as it will block the main thread
    #     Instead, consider running a separate process or a thread (like RQ worker) to perform the training
    #     :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
    #     :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
    #     """
    #
    #     # use cache to retrieve the data from the previous fit() runs
    #     old_data = self.get('my_data')
    #     old_model_version = self.get('model_version')
    #     print(f'Old data: {old_data}')
    #     print(f'Old model version: {old_model_version}')
    #
    #     # store new data to the cache
    #     self.set('my_data', 'my_new_data_value')
    #     self.set('model_version', 'my_new_model_version')
    #     print(f'New data: {self.get("my_data")}')
    #     print(f'New model version: {self.get("model_version")}')
    #
    #     print('fit() completed successfully.')
