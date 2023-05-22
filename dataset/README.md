Para obtener una copia del Dataset no es necesario clonar todo el repositorio. 

Clonar sólo el directorio _**dataset:**_
```bash
git clone -n --depth=1 --filter=tree:0 https://github.com/mauriciosierrav/cattle-detection.git
cd cattle-detection
git sparse-checkout set --no-cone dataset/labels dataset/images
git checkout
```

Opcional _(si quiere renombrar el directorio como "dataset_propio"):_
```bash
cd ..
mv cattle-detection/dataset dataset_propio
rm -rf cattle-detection/
```
