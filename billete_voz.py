import cv2
import os
import threading
import pyttsx3
from ultralytics import YOLO

# 1. Configuración de Voz
def hablar_en_hilo(texto):
    def hablar():
        tts = pyttsx3.init()
        tts.setProperty('rate', 160)
        tts.say(texto)
        tts.runAndWait()

    threading.Thread(target=hablar, daemon=True).start()

# 2. Carga del Modelo
script_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(script_dir, "models", "best.pt")
model = YOLO(modelo_path)

# DICCIONARIO DE FRASES DE VOZ (Usamos claves en minúsculas y limpias)
NOMBRES_LECTURA = {
    "5": "Billete de 5 euros",
    "10": "Billete de 10 euros",
    "20": "Billete de 20 euros",
    "50": "Billete de 50 euros",
    "100": "Billete de 100 euros",
    "200": "Billete de 200 euros"
}

cap = cv2.VideoCapture(0)

FRAMES_REQUERIDOS = 10  # Reducido a 10 fotogramas para una respuesta más rápida (~0.3 seg)
contador_estabilidad = 0
ultimo_detectado = None
ultimo_anunciado = None

print("¡Cajero con voz activo! Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.50)
    billetes_en_frame = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name_raw = model.names[cls_id]
        print(f"Billete detectado={class_name_raw}") # Ejemplos: '50_1', '50_2', '5_1', etc.
        
        # Filtro: Ignorar explícitamente manos o personas
        if class_name_raw.lower() in ["hand", "mano", "person", "persona"]:
            continue
            
        # Cortamos el texto en el primer '_' y nos quedamos con lo que hay a la izquierda
        # Ejemplo: '50_1' -> ['50', '1'] -> nos quedamos con '50'
        valor_billete = class_name_raw.split('_')[0].strip()
        print(f"Valor billete={valor_billete}")

        if valor_billete in NOMBRES_LECTURA:
            billetes_en_frame.append(valor_billete)

    # -------------------------------------------------------------
    # LÓGICA DE ESTABILIDAD
    # -------------------------------------------------------------
    if len(billetes_en_frame) > 0:
        billete_actual = billetes_en_frame[0]

        if billete_actual == ultimo_detectado:
            contador_estabilidad += 1
        else:
            contador_estabilidad = 1
            ultimo_detectado = billete_actual

        if contador_estabilidad >= FRAMES_REQUERIDOS and billete_actual != ultimo_anunciado:
            frase = NOMBRES_LECTURA[billete_actual]
            print(f"🔊 ANUNCIO DE VOZ: {frase}")
            hablar_en_hilo(frase)
            ultimo_anunciado = billete_actual

    else:
        contador_estabilidad = 0
        ultimo_detectado = None
        ultimo_anunciado = None

    # -------------------------------------------------------------
    # VISUALIZACIÓN
    # -------------------------------------------------------------
    annotated_frame = results[0].plot()

    progreso = min(1.0, contador_estabilidad / FRAMES_REQUERIDOS)
    ancho_barra = int(300 * progreso)
    
    cv2.rectangle(annotated_frame, (20, 20), (320, 45), (50, 50, 50), -1)
    color_barra = (0, 255, 0) if contador_estabilidad >= FRAMES_REQUERIDOS else (0, 255, 255)
    
    cv2.rectangle(annotated_frame, (20, 20), (20 + ancho_barra, 45), color_barra, -1)
    
    texto_estado = f"Estabilidad: {int(progreso*100)}%"
    if ultimo_detectado:
        texto_estado += f" ({ultimo_detectado} EUR)"
        
    cv2.putText(annotated_frame, texto_estado, (30, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    cv2.imshow("Detector de Billetes", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()