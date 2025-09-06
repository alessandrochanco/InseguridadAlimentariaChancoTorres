from modelo import ejecutar_modelo
from pathlib import Path
import time

def ejecutar():
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path("static") / f"run_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # Ejecutar el modelo y obtener las imágenes generadas
        images = ejecutar_modelo(str(outdir))
        return images  # Retorna las imágenes generadas
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    result = ejecutar()
    print(result)

    

