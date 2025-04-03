# Modulo para llamado de funciones utilitarias. Funciones para complementar las necesidades de los modelos desarrollados
from ollama import show as ollama_show

# Funcion para validacion de existencia de modelo solicitado
def is_model_available_locally(model_name: str) -> bool:
    """Funcion que recibe como parametro el nombre de un modelo de llm local, este valida su existencia en el computador
    en caso no se encuentre retorna una valor False, en caso contrario un valor True
    
    Arguments:
        - name_mode: Nombre del momdelo solicitado
    Return: 
        - exists: Valor booleano de la existencia del modelo
    """

    try:
        ollama_show(model_name)
        return True
    except Exception as e:
        print(f"No se encontro el modelo solicitado. {e}")
        return False
