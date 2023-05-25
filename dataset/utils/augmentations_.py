import albumentations as A
import cv2
import random
import numpy as np
import itertools
import os


class Albumentations:
    def __init__(self, path_img: str, path_label: str, augmentations: int):
        """
        If augmentations = 0 it will do all the pipelines that are configured in the Pipeline class.
        """
        image = cv2.imread(path_img)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes_r = np.loadtxt(path_label)
        self.bboxes = [list(np.append(i[1:], i[0])) for i in bboxes_r]

        H, W = self.image.shape[:2]
        geometric_transforms = [A.SafeRotate(p=1, always_apply=True),
                                A.Flip(p=1, always_apply=True)]
        crops_transforms = [A.RandomResizedCrop(height=H, width=W, p=1, always_apply=True)]
        sat_con_bri_hue = [A.ToGray(p=1, always_apply=True),
                           A.HueSaturationValue(p=1, always_apply=True),
                           A.RandomBrightnessContrast(p=1, always_apply=True)]
        effects_or_simulations = [A.RandomRain(p=1, always_apply=True, blur_value=1, brightness_coefficient=0.9),
                                  A.RandomShadow(p=1, always_apply=True, num_shadows_upper=100,
                                                 shadow_roi=(0, 0.3, 1, 1))]
        blur = [A.Blur(p=1, always_apply=True)]

        transforms1 = geometric_transforms + crops_transforms
        transforms2 = sat_con_bri_hue + effects_or_simulations + blur
        Transforms_ = [list(x) for x in itertools.product(transforms1, transforms2)]

        if augmentations == 0:
            self.Transforms = random.sample(Transforms_, len(Transforms_))
        elif augmentations > len(Transforms_):
            raise Exception(f'The maximum number of augmentations is {len(Transforms_)}')
        else:
            self.Transforms = random.sample(Transforms_, augmentations)

    def exec_pipeline(self) -> list:
        """
        Returns a list containing:
            a list with the original image and transformed images
            a list with the original labels and transformed labels,
            a list with the applied transformations.
        """
        images = [self.image]
        bboxes = [self.bboxes]
        transforms = ['Original']
        for T in self.Transforms:
            transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))
            transformed = transform(image=self.image, bboxes=self.bboxes)

            images.append(transformed['image'])
            bboxes.append(transformed['bboxes'])
            transforms.append([str(e).split("(")[0] for e in T])
        return [images, bboxes, transforms]


class AlbumentationsBatch:
    def __init__(self, path_imgs: str, path_labels: str, new_path_imgs: str,
                 new_path_labels: str, allowed_extensions: list, augmentations: int):
        self.path_imgs = path_imgs
        self.path_labels = path_labels
        self.new_path_imgs = new_path_imgs
        self.new_path_labels = new_path_labels
        self.allowed_extensions = allowed_extensions
        self.augmentations = augmentations
        self.files = [file for file in os.listdir(path_imgs)
                      if str.lower(os.path.splitext(file)[1]) in self.allowed_extensions]

    def exec_batch_pipeline(self):
        """
        """
        try:
            # iterate for each image and apply the defined number of augmentations
            for file in self.files:
                try: 
                    root, ext = os.path.splitext(file)
                    path_img = os.path.join(self.path_imgs, root + ext)
                    path_label = os.path.join(self.path_labels, root + '.txt')

                    imgs = Albumentations(path_img=path_img, path_label=path_label,
                                        augmentations=self.augmentations).exec_pipeline()
                    transformations = len(imgs[0])

                    for i in range(transformations):
                        # transformation name
                        t = '-'.join(imgs[2][i])
                        # save each transformed image
                        image = cv2.cvtColor(imgs[0][i], cv2.COLOR_RGB2BGR)
                        if imgs[2][i] == 'Original':
                            t = imgs[2][i]
                        cv2.imwrite(os.path.join(self.new_path_imgs, f'{root + "-" + t + ".jpg"}'), image)

                        # save each transformed label
                        with open(os.path.join(self.new_path_labels, f'{root + "-" + t + ".txt"}'), 'w') as f:
                            for label in imgs[1][i]:
                                label = [0] + list(label[:4])
                                line = ' '.join(str(e) for e in label)
                                f.write(line + '\n')
                except Exception as e:
                    print (f'image {path_img} failed with Exception:',e)
        except Exception as e:
            raise Exception(e)
        




    

