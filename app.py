from flask import Flask, render_template, jsonify
from ejecutar_modelo import ejecutar  # Importar la función desde ejecutar_modelo.py

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    try:
        # Llamar a la función que ejecuta el modelo y obtiene las imágenes
        images = ejecutar()

        # Verificar si ocurrió un error
        if isinstance(images, dict) and "error" in images:
            return jsonify({"ok": False, "error": images["error"]}), 500

        # Convertir las rutas de las imágenes a un formato adecuado para la web
        images_web = [p.replace('\\', '/') for p in images]

        # Pasar las imágenes a la plantilla (en formato JSON)
        return jsonify({"ok": True, "images": images_web})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
