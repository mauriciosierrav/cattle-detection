# Re-entrenar un modelo _YOLOv5_ en Google Colab
1. Abrir un nuevo cuaderno de Google Colab
   
    En la barra de menu seguir la siguiente ruta: `Entorno de ejecuión  >  Ver recursos > Cambiar tipo de entorno de ejecución > Acelerado por hardware`

    Seleccionar `GPU` y `Guardar`

    Esto permitirá acelerar el tiempo de entrenamiento y en general el rendimiento de todos los procesos
###
2.  Clonar repositorio YOLOv5 e instalar dependencias para el proyecto
    ```bash
    # clone repo yolov5
    !git clone https://github.com/ultralytics/yolov5
    # copy requirements.txt to source
    !cp /content/yolov5/requirements.txt /content/requirements.txt
    # install dependencies
    !pip install -U -r yolov5/requirements.txt
    !pip install fiftyone
    # Reinstall Pillow because the default version 9.4.0 generates compatibility errors.
    !pip install Pillow==9.0.0
    ```
    Dar click en <b>`RUNSTART RUNTIME`</b> una vez termine de correr el proceso.
###
3.  Verificar que si esté activa la GPU
    ```bash
    %cd yolov5
    import torch
    from yolov5 import utils 
    display = utils.notebook_init()
    ```
    Es probable que se asigne una GPU **Tesla T4, 15102MiB**
###
4.  Obtener imágenes de Open Images usando `fiftyone` para crear _**dataset_externo**_
    ```python
    import fiftyone as fo
    import fiftyone.zoo as foz
    
    splits = ["train", "validation", "test"]
    numSamples = 250
    seed = 42
        
    # Get {numSamples} images (maybe in total, maybe of each split) from fiftyone.  
    if fo.dataset_exists("open-images-cows"):
        fo.delete_dataset("open-images-cows")
    
    dataset = foz.load_zoo_dataset(
      "open-images-v7",
      splits=splits,
      label_types=["detections"],
      classes="Cattle",
      max_samples=numSamples,
      seed=seed,
      shuffle=True,
      dataset_name="open-images-cows"
      )
        
    # Take a quick peek to see what's there
    print(dataset)
    ```
###
5.  Crear directorio _**dataset_externo**_ con la estrcutura necesaria para YOLOv5
    ```python
    %cd ..
    export_dir = "dataset_externo/"
    label_field = "ground_truth"
        
    # The splits to export
    splits = ["train", "validation","test"]
        
    # All splits must use the same classes list
    classes = ["Cattle"]
        
    # The dataset or view to export
    # We assume the dataset uses sample tags to encode the splits to export
    dataset_or_view = fo.load_dataset("open-images-cows")
    
    # Export the splits
    for split in splits:
        split_view = dataset_or_view.match_tags(split)
        split_view.export(
          export_dir=export_dir,
          dataset_type=fo.types.YOLOv5Dataset,
          label_field=label_field,
          split=split,
          classes=classes,
          )
    ```

    El _**dataset_externo**_ quedará en la siguiente estrcutura.
    ```
    content ──── dataset_externo
                       ├───────── images/  
                       ├             ├── train/         
                       ├             ├── validation/
                       ├             └── test/
                       ├───────── labels/  
                       ├              ├── train/         
                       ├              ├── validation/
                       ├              └── test/
                       ├───────── dataset.yml
    ```
###
6.  Modificar _dataset.yml_, dado que el código de entrenamiento espera la etiqueta `val` en lugar de `validation`.
    ```python
    import yaml
       
    with open('dataset_externo/dataset.yaml') as f:
      try:
        doc = yaml.safe_load(f)
        doc['val'] = doc.pop('validation')
      except yaml.YAMLError as exc:
        print(exc)
       
    with open('dataset_externo/dataset.yaml', 'w') as f:
      try:
        yaml.dump(doc, f)
      except yaml.YAMLError as exc:
        print(exc)
    ```
###
7.  Clonar repositorio que tiene el _**dataset_propio**_
    ```bash
    %%shell
    file_ID="1VffSYJMzuC-uq9sv2flhxPdwYkXqOxR9"
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$file_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_ID" -O dataset_propio.zip &>/dev/null
    rm -rf /tmp/cookies.txt
    unzip dataset_propio.zip  &>/dev/null
    rm -rf dataset_propio.zip
    echo 'Proceso finalizado correctamente'
    
    # copy dataset.yaml from dataset_externo
    cp /content/dataset_externo/dataset.yaml /content/dataset_propio
    ```
    
    El _**dataset_propio**_ se clonará en la siguiente estrcutura.
    ```
    content ──── dataset_propio
                       ├───────── images/  
                       ├             ├── train/         
                       ├             ├── val/
                       ├             └── test/
                       ├───────── labels/  
                       ├              ├── train/         
                       ├              ├── val/
                       ├              └── test/
                       ├───────── dataset.yml
    ```
###
8.  Copiar los siguientes archivos que contienen cambios necesarios respecto a los valores predeterminados de _yolov5_
    ```bash
    ! cp /content/dataset_propio/utils/custom_yolo_files/augmentations.py /content/yolov5/utils/
    ! cp /content/dataset_propio/utils/custom_yolo_files/hyp.scratch-low.yaml /content/yolov5/data/hyps/
    ! cp /content/dataset_propio/utils/custom_yolo_files/val_custom.py /content/yolov5/
    ```
###
9.  Modificar _dataset.yml_, dado que `path` y `val` no serían correctos para el _**dataset_propio**_
    ```python
    import yaml
       
    with open('dataset_propio/dataset.yaml') as f:
      try:
        doc = yaml.safe_load(f)
        doc['path'] = '/content/dataset_propio'
        doc['val'] = './images/val'
      except yaml.YAMLError as exc:
        print(exc)
       
    with open('dataset_propio/dataset.yaml', 'w') as f:
      try:
        yaml.dump(doc, f)
      except yaml.YAMLError as exc:
        print(exc)
    ```
###
10. Crear los directorios `dataset_propio_aug` y `dataset_externo_aug` con sus respectivos sub-directorios para almacenar las imagenes aumentadas.
    ```bash
    !mkdir -p ./dataset_propio_aug/images/train/
    !mkdir -p ./dataset_propio_aug/labels/train/
    !mkdir -p ./dataset_externo_aug/images/train/
    !mkdir -p ./dataset_externo_aug/labels/train/
    ```
###
11.  Crear nuevas imágenes para `train` del _**dataset_propio**_ utilizando **Albumentations**.
  ```python
    from dataset_propio.utils import AlbumentationsBatch
        
    # define mandatory variables
    path_imgs = './dataset_propio/images/train/'  # path of images to be transformed
    path_labels = './dataset_propio/labels/train/'  # path of labels to be transformed
    new_path_imgs = './dataset_propio_aug/images/train/'  # path where the transformed images will be stored
    new_path_labels = './dataset_propio_aug/labels/train/'  # path where the transformed labels will be stored
        
    batch = AlbumentationsBatch(path_imgs, path_labels, new_path_imgs, new_path_labels,
                                    allowed_extensions=['.jpg', '.jpeg', '.png'], augmentations=5)
    batch.exec_batch_pipeline()
  ``` 
###
12. Crear nuevas imágenes para `train` del _**dataset_externo**_ utilizando **Albumentations**
    ```python
    from dataset_propio.utils import AlbumentationsBatch
    
    # define mandatory variables
    path_imgs = './dataset_externo/images/train/'  # path of images to be transformed
    path_labels = './dataset_externo/labels/train/'  # path of labels to be transformed
    new_path_imgs = './dataset_externo_aug/images/train/'  # path where the transformed images will be stored
    new_path_labels = './dataset_externo_aug/labels/train/'  # path where the transformed labels will be stored
        
    batch = AlbumentationsBatch(path_imgs, path_labels, new_path_imgs, new_path_labels,
                                    allowed_extensions=['.jpg', '.jpeg', '.png'], augmentations=5)
    batch.exec_batch_pipeline()
    ```
###
13. Considerando que el dataset de validación será siempre el mismo para todos los modelos se deben ejecutar los siguientes comandos.
    ```bash
    !mkdir -p ./dataset_propio/images/test/
    !mkdir -p ./dataset_propio/labels/test/

    !cp ./dataset_propio/images/val/* ./dataset_propio/images/test/
    !cp ./dataset_propio/labels/val/* ./dataset_propio/labels/test/

    !rm ./dataset_externo/images/test/*
    !rm ./dataset_externo/images/validation/*
    !rm ./dataset_externo/labels/test/*
    !rm ./dataset_externo/labels/validation/*

    !cp ./dataset_propio/images/test/* ./dataset_externo/images/test/
    !cp ./dataset_propio/images/val/* ./dataset_externo/images/validation/
    !cp ./dataset_propio/labels/test/* ./dataset_externo/labels/test/
    !cp ./dataset_propio/labels/val/* ./dataset_externo/labels/validation/

    !mkdir -p ./dataset_externo_aug/images/test/
    !mkdir -p ./dataset_externo_aug/images/validation/
    !mkdir -p ./dataset_externo_aug/labels/test/
    !mkdir -p ./dataset_externo_aug/labels/validation/

    !cp ./dataset_propio/images/test/* ./dataset_externo_aug/images/test/
    !cp ./dataset_propio/images/val/* ./dataset_externo_aug/images/validation/
    !cp ./dataset_propio/labels/test/* ./dataset_externo_aug/labels/test/
    !cp ./dataset_propio/labels/val/* ./dataset_externo_aug/labels/validation/

    !mkdir -p ./dataset_propio_aug/images/test/
    !mkdir -p ./dataset_propio_aug/images/val/
    !mkdir -p ./dataset_propio_aug/labels/test/
    !mkdir -p ./dataset_propio_aug/labels/val/

    !cp ./dataset_propio/images/test/* ./dataset_propio_aug/images/test/
    !cp ./dataset_propio/images/val/* ./dataset_propio_aug/images/val/
    !cp ./dataset_propio/labels/test/* ./dataset_propio_aug/labels/test/
    !cp ./dataset_propio/labels/val/* ./dataset_propio_aug/labels/val/

###
14. Configurar los archivos _dataset.yaml_ para los datasets aumentados
    ```bash
    !cp /content/dataset_externo/dataset.yaml /content/dataset_externo_aug
    !cp /content/dataset_propio/dataset.yaml /content/dataset_propio_aug
      ```
    ```python
    with open('dataset_propio_aug/dataset.yaml') as f:
      try:
        doc = yaml.safe_load(f)
        doc['path'] = '/content/dataset_propio_aug'
      except yaml.YAMLError as exc:
        print(exc)
    
    with open('dataset_propio_aug/dataset.yaml', 'w') as f:
      try:
        yaml.dump(doc, f)
      except yaml.YAMLError as exc:
        print(exc)
      ```
      ```python
    with open('dataset_externo_aug/dataset.yaml') as f:
      try:
        doc = yaml.safe_load(f)
        doc['path'] = '/content/dataset_externo_aug'
      except yaml.YAMLError as exc:
        print(exc)
      
    with open('dataset_externo_aug/dataset.yaml', 'w') as f:
      try:
        yaml.dump(doc, f)
      except yaml.YAMLError as exc:
        print(exc)
      ```
###
15. Realizar re-entrenamiento al modelo _yolov5s.pt_ (o cualquier otro modelo de YOLOv5) usando los dataset originales y/o aumentados.
    ```bash
    # re-entrenamiento con el dataset_externo
    !python yolov5/train.py --weights yolov5s.pt --data /content/dataset_externo/dataset.yaml --epochs 50 --batch 10 --project train/ --name dataset_externo
    ```
    ```bash
    # re-entrenamiento con el dataset_propio
    !python yolov5/train.py --weights yolov5s.pt --data /content/dataset_propio/dataset.yaml --epochs 50 --batch 10 --project train/ --name dataset_propio
    ```
    ```bash
    # re-entrenamiento al modelo dataset_externo_best.pt con el dataset_propio
    !python yolov5/train.py --weights /content/train/dataset_externo/weights/best.pt --data /content/dataset_propio/dataset.yaml --epochs 50 --batch 10 --project train/ --name dataset_propio2
    ```
    ```bash
    # re-entrenamiento con el dataset_externo aumentado
    !python yolov5/train.py --weights yolov5s.pt --data /content/dataset_externo_aug/dataset.yaml --epochs 50 --batch 10 --project train/ --name dataset_externo_aug
    ```
    ```bash
    # re-entrenamiento con el dataset_propio aumentado
    !python yolov5/train.py --weights yolov5s.pt --data /content/dataset_propio_aug/dataset.yaml --epochs 50 --batch 10 --project train/ --name dataset_propio_aug
    ```    
    ```bash
    # re-entrenamiento al modelo dataset_externo_aug_best.pt con el dataset_propio ambos aumentados
    !python yolov5/train.py --weights /content/train/dataset_externo_aug/weights/best.pt --data /content/dataset_propio_aug/dataset.yaml --epochs 50 --batch 10 --project train/ --name dataset_propio_aug2
    ```

    Los argumentos más importantes son:
    * --weights: <font color="Green">model path or triton URL</font>
    * --data: <font color="Green">dataset.yaml path</font> 
    * --hyp: <font color="Green">default</font> = `/content/yolov5/data/hyps/hyp.scratch-low.yaml'`
    * --epochs: 50
    * --batch: 10
    * --img: <font color="Green">default</font> = 640
    * --workers: <font color="Green">default</font> = 8
    * --project: train/
    * --name: save results to project/<font color="Green">**name**</font>
###
16. <span style="color: Red; "> Descargar los modelos seleccionados y llevarlos a la carpeta fastapi/model/ para usarlos en la API.
###
17. Ejecutar la inferencia de detección de objetos en los diferentes modelos (opcional)
    ```bash
    # Detect con el modelo pre-entrenado yolov5s.pt
    !python yolov5/detect.py --weights yolov5s.pt --data /content/yolov5/data/coco128.yaml --source /content/dataset_propio/images/test/ --conf-thres 0.4 --project detect/ --name yolov5
    ```
    ```bash
    # Detect con el modelo yolov5s.pt re-entrenado con el dataset_propio  
    !python yolov5/detect.py --weights /content/train/dataset_propio/weights/best.pt --data /content/dataset_propio/dataset.yaml --source /content/dataset_propio/images/test/ --conf-thres 0.4 --project detect/ --name dataset_propio
    ```
    Los argumentos más importantes son:
    * --weights: <font color="Green">model path or triton URL</font>
    * --data: <font color="Green">dataset.yaml path</font> 
    * --source: `/content/dataset_propio/images/test/`
    * --conf-thres: 0.4
    * --project: detect/
    * --name: save results to project/<font color="Green">**name**</font>
###
18. Probar el desempeño del modelo pre-entrenado Yolo sobre el dataset de validación para tener un desempeño de referencia. 
    ```bash
    !mkdir -p ./dataset_propio_validation/
    !cp -r /content/dataset_propio/* /content/dataset_propio_validation/
    !cp /content/yolov5/data/coco128.yaml /content/dataset_propio_validation/dataset.yaml
    ```
    ```python
    import yaml

    with open('dataset_propio_validation/dataset.yaml') as f:
      try:
        doc = yaml.safe_load(f)
        doc['path'] = '/content/dataset_propio_validation'
        doc['val'] = './images/val'
        doc['test'] = './images/test/'
        doc['train'] = './images/train/'
        doc['download'] = ''
      except yaml.YAMLError as exc:
        print(exc)

    with open('dataset_propio_validation/dataset.yaml', 'w') as f:
      try:
        yaml.dump(doc, f)
      except yaml.YAMLError as exc:
        print(exc)

    import os
    path_to_edit = '/content/dataset_propio_validation/labels/val/'
    images = os.listdir(path_to_edit)
    img_paths = [path_to_edit + img for img in images]
    
    def replace_class(txt_file):
      replaced_content = ""
      with open(txt_file, 'r') as f:
        for line in f.readlines():
          new_line = '19'+line[1:]
          replaced_content = replaced_content + new_line
      f.close()

    write_file = open(txt_file, "w")
    write_file.write(replaced_content)
    write_file.close()

    for txt_path in img_paths:
      replace_class(txt_path)
    ```
    ```bash
    ! mkdir -p dataset_propio_validation/images/test/
    ! mkdir -p dataset_propio_validation/labels/test/
    ! cp dataset_propio_validation/images/val/* dataset_propio_validation/images/test/
    ! cp dataset_propio_validation/labels/val/* dataset_propio_validation/labels/test/

    # Validation   
    python yolov5/val_custom.py --weights yolov5s.pt --data /content/dataset_propio_validation/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_propio
    ```
###
19. Medir la robustez del modelo generando datasets que reflejen condiciones como nieve, noche, cambios de perpectiva, baja resolución, niebla, entre otros, y evaluando el desempeño del modelo respecto a cada uno de ellos. 

    ```bash
    !mkdir -p ./dataset_robustez_downscale/images/val/
    !mkdir -p ./dataset_robustez_downscale/labels/val/

    !mkdir -p ./dataset_robustez_randomfog/images/val/
    !mkdir -p ./dataset_robustez_randomfog/labels/val/

    !mkdir -p ./dataset_robustez_randomsnow/images/val/
    !mkdir -p ./dataset_robustez_randomsnow/labels/val/

    !mkdir -p ./dataset_robustez_night/images/val/
    !mkdir -p ./dataset_robustez_night/labels/val/

    !mkdir -p ./dataset_robustez_mud/images/val/
    !mkdir -p ./dataset_robustez_mud/labels/val/
    ```
    ```python
    from dataset_propio.utils import AlbumentationsBatch,TransformsPipeline
    import albumentations as A

    def generate_sturdiness_dataset(transform,source_folder,destination_folder):
        transforms_robustez = TransformsPipeline(transforms1 = transform, augmentations = 1)
        print('generating pipeline',transforms_robustez.transforms_pipeline)
        path_imgs = source_folder + 'images/val/'
        path_labels = source_folder + 'labels/val/'
        new_path_imgs = destination_folder + 'images/val/'
        new_path_labels = destination_folder + 'labels/val/'
        batch = AlbumentationsBatch(path_imgs, path_labels, new_path_imgs, new_path_labels,
                                    allowed_extensions=['.jpg', '.jpeg', '.png'], 
                                    transforms_pipeline = transforms_robustez,
                                    augmentations=1).exec_batch_pipeline()

    sturdiness_transforms = [[A.Downscale(p=1, always_apply=True,scale_min=0.2, scale_max=0.2)],
                            [A.RandomFog(p=1, always_apply=True,fog_coef_upper=0.3,fog_coef_lower=0.3)],
                            [A.RandomSnow(p=1, always_apply=True,snow_point_lower=0.3, snow_point_upper=0.3,
                                brightness_coeff=3.5)],
                            [A.ColorJitter(p=1, always_apply=True,brightness=(0.3,0.3), 
                                  contrast=(0.5,0.5), 
                                  saturation=(0.6,0.6), 
                                  hue=(0.3,0.3))],                         
                            [A.Perspective(p=1, always_apply=True,scale=(0.2, 0.2), keep_size=True, 
                                  interpolation=1,fit_output=True)]]
    sturdiness_folders = ['./dataset_robustez_downscale/','./dataset_robustez_randomfog/',
                          './dataset_robustez_randomsnow/','./dataset_robustez_night/',
                          './dataset_robustez_mud/']
    for t,f in zip(sturdiness_transforms,sturdiness_folders):
      generate_sturdiness_dataset(t,'./dataset_propio/',f)
    ```
    ```bash
    !rm ./dataset_robustez_downscale/images/val/*Original.jpg
    !rm ./dataset_robustez_downscale/labels/val/*Original.txt
    !rm ./dataset_robustez_randomfog/images/val/*Original.jpg
    !rm ./dataset_robustez_randomfog/labels/val/*Original.txt
    !rm ./dataset_robustez_randomsnow/images/val/*Original.jpg
    !rm ./dataset_robustez_randomsnow/labels/val/*Original.txt
    !rm ./dataset_robustez_night/images/val/*Original.jpg
    !rm ./dataset_robustez_night/labels/val/*Original.txt
    !rm ./dataset_robustez_mud/images/val/*Original.jpg
    !rm ./dataset_robustez_mud/labels/val/*Original.txt

    ! mkdir -p dataset_robustez_downscale/images/test/
    ! mkdir -p dataset_robustez_downscale/labels/test/
    ! cp dataset_robustez_downscale/images/val/* dataset_robustez_downscale/images/test/
    ! cp dataset_robustez_downscale/labels/val/* dataset_robustez_downscale/labels/test/
    ! cp dataset_propio/dataset.yaml dataset_robustez_downscale/


    ! mkdir -p dataset_robustez_randomfog/images/test/
    ! mkdir -p dataset_robustez_randomfog/labels/test/
    ! cp dataset_robustez_randomfog/images/val/* dataset_robustez_randomfog/images/test/
    ! cp dataset_robustez_randomfog/labels/val/* dataset_robustez_randomfog/labels/test/
    ! cp dataset_propio/dataset.yaml dataset_robustez_randomfog/


    ! mkdir -p dataset_robustez_randomsnow/images/test/
    ! mkdir -p dataset_robustez_randomsnow/labels/test/
    ! cp dataset_robustez_randomsnow/images/val/* dataset_robustez_randomsnow/images/test/
    ! cp dataset_robustez_randomsnow/labels/val/* dataset_robustez_randomsnow/labels/test/
    ! cp dataset_propio/dataset.yaml dataset_robustez_randomsnow/


    ! mkdir -p dataset_robustez_night/images/test/
    ! mkdir -p dataset_robustez_night/labels/test/
    ! cp dataset_robustez_night/images/val/* dataset_robustez_night/images/test/
    ! cp dataset_robustez_night/labels/val/* dataset_robustez_night/labels/test/
    ! cp dataset_propio/dataset.yaml dataset_robustez_night/


    ! mkdir -p dataset_robustez_mud/images/test/
    ! mkdir -p dataset_robustez_mud/labels/test/
    ! cp dataset_robustez_mud/images/val/* dataset_robustez_mud/images/test/
    ! cp dataset_robustez_mud/labels/val/* dataset_robustez_mud/labels/test/
    ! cp dataset_propio/dataset.yaml dataset_robustez_mud/

    #Se deben editar los dataset.yaml para que apunten al folder correcto

    #Correr las validaciones:
    python yolov5/val.py --weights best_externo.pt --data /content/dataset_robustez_downscale/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_downscale_ext

    python yolov5/val.py --weights best_externo.pt --data /content/dataset_robustez_randomfog/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_randomfog_ext

    python yolov5/val.py --weights best_externo.pt --data /content/dataset_robustez_randomsnow/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_randomsnow_ext

    python yolov5/val.py --weights best_externo.pt --data /content/dataset_robustez_night/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_night_ext

    python yolov5/val.py --weights best_externo.pt --data /content/dataset_robustez_mud/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_mud_ext


    python yolov5/val.py --weights best_externo_aug.pt --data /content/dataset_robustez_downscale/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_downscale_ext_aug

    python yolov5/val.py --weights best_externo_aug.pt --data /content/dataset_robustez_randomfog/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_randomfog_ext_aug

    python yolov5/val.py --weights best_externo_aug.pt --data /content/dataset_robustez_randomsnow/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_randomsnow_ext_aug

    python yolov5/val.py --weights best_externo_aug.pt --data /content/dataset_robustez_night/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_night_ext_aug

    python yolov5/val.py --weights best_externo_aug.pt --data /content/dataset_robustez_mud/dataset.yaml --task test --conf-thres 0.5 --project Validation/ --name dataset_robustez_mud_ext_aug
    ```
#
## Fuentes de consulta

1. https://www.codeproject.com/Articles/5347827/How-to-Train-a-Custom-YOLOv5-Model-to-Detect-Objec

2. https://www.analyticsvidhya.com/blog/2021/08/train-your-own-yolov5-object-detection-model/

3. https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208

4. https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843

5. https://github.com/ultralytics/yolov5

6. https://www.analyticsvidhya.com/blog/2023/02/how-to-train-a-custom-dataset-with-yolov5/

7. https://machinelearningknowledge.ai/introduction-to-yolov5-object-detection-with-tutorial/#v_Example_of_YOLOv5l
