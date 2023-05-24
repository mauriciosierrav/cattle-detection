import torch
import fileinput

if __name__ == "__main__":
    torch.hub.load("ultralytics/yolov5", "yolov5s")
    torch.hub.load("ultralytics/yolov5", "yolov5l")

    # Modify the color for the bounding boxes
    '''
    hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    '''

    change = 0
    try:
        with fileinput.FileInput('./fastapi/yolov5/utils/plots.py', inplace=True) as archivo:
            for linea in archivo:
                if 'FF3838' in linea:
                    linea = linea.replace('FF3838', '6473FF')
                    change += 1
                print(linea, end='')

        with fileinput.FileInput('./fastapi/yolov5/utils/plots.py', inplace=True) as archivo:
            for linea in archivo:
                if 'FF37C7' in linea:
                    linea = linea.replace('FF37C7', '6473FF')
                    change += 1
                print(linea, end='')

            print(f'Changes in ./fastapi/yolov5/utils/plots.py: {change}')
    except Exception as e:
        raise Exception(e)
