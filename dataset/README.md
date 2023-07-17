Para obtener las _images_ y _labels_ es necesario seguir cualquiera de estas indicaciones: 

Usando la terminal
```bash
file_ID="1VffSYJMzuC-uq9sv2flhxPdwYkXqOxR9"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$file_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_ID" -O dataset_propio.zip &>/dev/null
rm -rf /tmp/cookies.txt
unzip dataset_propio.zip  &>/dev/null
rm -rf dataset_propio.zip
echo 'Proceso finalizado correctamente'
```

Usando python
```python
import requests, zipfile, io

file_ID = '1VffSYJMzuC-uq9sv2flhxPdwYkXqOxR9'

# URL del archivo a descargar
url = 'https://docs.google.com/uc?export=download&id={file_ID}'
# Realizar la primera solicitud para obtener las cookies
response = requests.get(url, stream=True)
cookies = response.cookies
# Extraer el valor del parámetro "confirm" de la respuesta
confirm_param = response.content.decode('utf-8').split(';')[-1].strip().split('=')[-1]
# Construir la URL final con las cookies y el valor "confirm"
final_url = f'https://docs.google.com/uc?export=download&confirm={confirm_param}&id={file_ID}'

# Obtener el archivo
r = requests.get(final_url, cookies=cookies, stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
print('Proceso finalizado correctamente')
```
