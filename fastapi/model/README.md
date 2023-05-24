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
    git clone -n --depth=1 --filter=tree:0 https://github.com/mauriciosierrav/cattle-detection.git
    cd cattle-detection
    git sparse-checkout set --no-cone dataset/labels dataset/images dataset/utils
    git checkout
    
    # replace name and delete cattle-detection folder
    cd ..
    mv cattle-detection/dataset dataset_propio
    rm -rf cattle-detection/
    
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
8.  Modificar _dataset.yml_, dado que `path` y `val` no serían correctos para el _**dataset_propio**_
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
9.  Crear nuevas imágenes para `train` del _**dataset_propio**_ utilizando **Albumentations**
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
10. Crear nuevas imágenes para `train` del _**dataset_externo**_ utilizando **Albumentations**
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
11. <span style="color: Red; "> Borrar lo de hyps y lo de augmentation de YOLOv5 </span>
###
12. Realizar re-entrenamiento al modelo _yolov5s.pt_ (o cualquier otro modelo de YOLOv5)
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
13. <span style="color: Red; "> Realizar <code>`test`</code> de cada uno de los modelos para obtener métricas de desempeño y compararlos entré sí (si es necesario? no es suficiente con el val que hace por defecto al entrenar? si realmente es necesario hay que dejar 5 imgs en val y la sotras reservarlas para este punto.</span>
###
14. <span style="color: Red; "> generar aumentos en el dataset de val para correr nuevamente val y probar la robustez del modelo</span>
###
15. <span style="color: Red; "> Descargar los modelos seleccionados y llevarlos a la carpeta fastapi/model/ para usarlos en la API.
###
16. Ejecutar la inferencia de detección de objetos en los diferentes modelos (opcional)
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

#
## Fuentes de consulta

1. https://www.codeproject.com/Articles/5347827/How-to-Train-a-Custom-YOLOv5-Model-to-Detect-Objec

2. https://www.analyticsvidhya.com/blog/2021/08/train-your-own-yolov5-object-detection-model/

3. https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208

4. https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843

5. https://github.com/ultralytics/yolov5

6. https://www.analyticsvidhya.com/blog/2023/02/how-to-train-a-custom-dataset-with-yolov5/

7. https://machinelearningknowledge.ai/introduction-to-yolov5-object-detection-with-tutorial/#v_Example_of_YOLOv5l
