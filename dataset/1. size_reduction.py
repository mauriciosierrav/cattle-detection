'''
No es necesario usarlo ya que YoloV5 redimensiona las imágenes de manera predeterminada
'''

from PIL import Image
import os


def image_size_reduction(curr_path: str, new_path: str, quality: int, target_width: int = 1000):
    image_ = Image.open(curr_path)

    # Resize the image
    width, height = image_.size
    target_height = int(target_width * height / width)
    new_size = (target_width, target_height)
    if target_width < width:
        resized_image = image_.resize(new_size)
        # Save the reduced image
        resized_image.save(new_path, optimize=True, quality=quality)


if __name__ == '__main__':
    images = os.listdir('./images/full/')
    # Iterate over all images
    for image in images:
        curr_path_ = f"./images/full/{image}"
        new_path_ = f"./images/full/{image}"
        image_size_reduction(curr_path_, new_path_, quality=100, target_width=1000)
