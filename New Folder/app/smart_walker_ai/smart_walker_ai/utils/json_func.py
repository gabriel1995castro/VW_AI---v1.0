# Librerias para lectura de archivos json
import json

def get_json_info(jsonPath: str):
    data = ""
    try:
        with open(jsonPath, 'r', encoding = 'utf-8') as file:
            data = json.load(file) 
    except Exception as e:
        print(f"Error en la lectura del archivo json. {e}")
    return data

