# cajero-ia
Proyecto de lectura de billetes y monedas de euro

# Instalación y ejecución de la app

##1. Entorno virtual Python
Inicia el entorno virtual de tu ordenador

##2. Librerías Python
Dile a tu Python que instale las librerías del fichero requirements.txt
```bash
pip install -r requirements.txt
```

## 3. Credenciales necesarias
Este proyecto requiere conexión a las APIs de IA y Google Sheets. Debes crear tus propios archivos:
- Crea un archivo `.env` basándote en la plantilla `.env.example` y rellena con tus datos.
- Descarga tus claves de Google Drive y guárdalas en la carpeta principal con el nombre `credentials.json`.

##4. Ejecución de la app
Ejecuta el archivo principal:
```bash
python cajero-ia.py 
```
