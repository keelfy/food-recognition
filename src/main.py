import argparse
import os

from utils import *


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i',
                    '--image',
                    required=True,
                    help='Path to the source image')
    ap.add_argument('-cl',
                    '--classes',
                    help='Path to text file containing class names',
                    default='yolov4.classes')
    ap.add_argument('-c',
                    '--config',
                    help='Path to text file with YOLO configuration',
                    default='yolov4.cfg')
    ap.add_argument('-w',
                    '--weights',
                    help='Path to pre-trained weights file',
                    default='yolov4.weights')
    return ap.parse_args()


def check_paths(arguments):
    if not os.path.isfile(arguments.image):
        print('Source image not found!')
        exit(-1)

    if not os.path.isfile(arguments.weights):
        print('YOLOv4 weights not found!')
        exit(-1)

    if not os.path.isfile(arguments.config):
        print('YOLOv4 configuration not found!')
        exit(-1)

    if not os.path.isfile(arguments.classes):
        print('YOLOv4 classes not found!')
        exit(-1)


def read_image(image_path):
    return cv2.imread(image_path)


def read_classes(classes_path):
    with open(classes_path, 'r') as f:
        result = [line.strip() for line in f.readlines()]
    return result


def show_and_save_image(img):
    cv2.imshow('Food Recognition', img)
    cv2.waitKey()
    cv2.imwrite("recognition-result.jpg", img)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Получаем аргументы из консоли
    args = get_arguments()
    # Проверяем существование файлов
    check_paths(args)

    # Загружаем изображение в память
    image = read_image(args.image)
    # Считываем классы
    classes = read_classes(args.classes)

    # Создаем предобученную нейронную сеть
    network = Network(args.weights, args.config)
    # Осуществляем предсказание содержимого изображения
    outs = network.predict(image)

    # Создаем класс, отрисовывающий результат обработки нейросетью
    drawer = PredictionDrawer(prediction_result=outs, classes=classes)
    # Отрисовываем результат на исходном изображении
    drawer.draw(image=image, image_width=image.shape[1], image_height=image.shape[0])

    # Открываем окно с результатом и сохраняем в файл
    show_and_save_image(img=image)
