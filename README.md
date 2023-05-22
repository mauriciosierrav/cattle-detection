![GitHub language count](https://img.shields.io/github/languages/count/mauriciosierrav/cattle-detection?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/mauriciosierrav/cattle-detection?style=plastic)
![GitHub repo size](https://img.shields.io/github/repo-size/mauriciosierrav/cattle-detection?style=plastic)

# Cattle Detection [API y WebApp]

## Comenzando 🚀

Siguiendo estas instrucciones podrás obtener una copia del proyecto en funcionamiento en tu máquina local para propósitos de desarrollo y pruebas.

1. Clonar el repositorio
    ```bash
    git clone git@github.com:mauriciosierrav/cattle-detection.git
    ```
####
2. Asignar un intérprete al proyecto _(se recomienda Python 3.10)_. Esto lo puede hacer con un intérprete local o creando y activando un ambiente virtual con su herramienta favorita para gestión de entornos virtuales.
####
3. Instalar requerimientos
    ```bash
    pip install -r requirements.txt 
    ```
####
4. Cambiar al directorio `fastapi` y clonar repositorio de `yolov5`
    ```bash
    cd fastapi ; git clone git@github.com:ultralytics/yolov5.git 
    ```
####
5. Cambiar al directorio principal del proyecto y ejecutar `main.py`
    ```bash
    cd .. ; python3 main.py
    ```
####
6. Mover los pesos descargados a la carpeta `fastapi/models`
    ```bash
    mv yolov5l.pt fastapi/model/ ; mv yolov5s.pt fastapi/model/
    ```
####
7. Cambiar al directorio `fastapi` y ejecutar `api.py` usando `uvicorn`
    ```bash
    cd fastapi ; uvicorn api:app --reload --host 127.0.0.1 --port 8000
    ```
8. ***En una nueva terminal***: cambiar al directorio `streamlit` y ejecutar `webapp.py`
    ```bash
    cd streamlit ; streamlit run webapp.py --server.port 8501 
    ```


## Autores ✒️

* **Juan Mauricio Sierra Valencia** - [mauriciosierrav](https://github.com/mauriciosierrav)
* **Vanessa Alexandra Lopera Mazo** - [vanesalo10](https://github.com/vanesalo10)

## License 📄

This software is [MIT](https://mit-license.org/) licensed

