import cv2
import os
import threading
import pyttsx3
from ultralytics import YOLO

# 1. Configuración del sintetizador de voz (Offline)
def hablar_en_hilo(texto):
    def hablar():
        tts = pyttsx3.init()
        tts.setProperty('rate', 160) # Velocidad de voz
        tts.say(texto)
        tts.runAndWait()

    threading.Thread(target=hablar, daemon=True).start()

# 2. Carga del Modelo Unificado
script_dir = os.path.dirname(os.path.abspath(__file__))
# Asegúrate de colocar aquí la ruta correcta a tu nuevo 'best.pt'
modelo_path = os.path.join(script_dir, "models", "best.pt") 
model = YOLO(modelo_path)

# 3. DICCIONARIO MAESTRO: Mapeo de identificadores a (Frase de Voz, Valor Numérico en €)
CATALOGO_DINERO = {
    # --- BILLETES ---
    "5": ("Billete de 5 euros", 5.0),
    "10": ("Billete de 10 euros", 10.0),
    "20": ("Billete de 20 euros", 20.0),
    "50": ("Billete de 50 euros", 50.0),
    "100": ("Billete de 100 euros", 100.0),
    "200": ("Billete de 200 euros", 200.0),
    "500": ("Billete de 500 euros", 500.0),

    # --- MONEDAS DE CÉNTIMO ---
    "1_cent": ("Moneda de 1 céntimo", 0.01),
    "2_cent": ("Moneda de 2 céntimos", 0.02),
    "5_cent": ("Moneda de 5 céntimos", 0.05),
    "10_cent": ("Moneda de 10 céntimos", 0.10),
    "20_cent": ("Moneda de 20 céntimos", 0.20),
    "50_cent": ("Moneda de 50 céntimos", 0.50),
    "100_cent": ("Moneda de 1 euro", 1.0),
    "200_cent": ("Moneda de 2 euros", 2.0)
}

def normalizar_clase(class_name_raw):
    """
    Traductor de nombres de clase del modelo al formato estándar del diccionario.
    Diferencia billetes ('50_1' -> '50') de monedas ('100' -> '100_cent')
    """
    raw = class_name_raw.lower().strip()

    # 1. Descartar manos o personas
    if raw in ["hand", "mano", "person", "persona"]:
        return None

    # 2. Caso Billetes: Contienen guion bajo de cara (ej: '50_1', '50_2')
    if "_" in raw:
        valor_billete = raw.split("_")[0]  # Nos quedamos con '50'
        if valor_billete in CATALOGO_DINERO:
            return valor_billete

    # 3. Caso Monedas: Es solo el número en céntimos (ej: '1', '10', '50', '100')
    if raw.isdigit():
        clave_moneda = f"{raw}_cent"  # Convertimos '100' -> '100_cent'
        if clave_moneda in CATALOGO_DINERO:
            return clave_moneda

    return None

# Código Principal
cap = cv2.VideoCapture(0)

FRAMES_REQUERIDOS = 12 # Umbral para validar la estabilidad
contador_estabilidad = 0
ultimo_detectado = None
ultimo_anunciado = None

print("¡Cajero Inteligente (Billetes y Monedas) Activo! Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizamos inferencia (0.60 para filtrar detecciones dudosas de monedas pequeñas)
    results = model(frame, conf=0.60)
    elementos_validos = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name_raw = model.names[cls_id]

        clase_normalizada = normalizar_clase(class_name_raw)
        
        if clase_normalizada and clase_normalizada in CATALOGO_DINERO:
            elementos_validos.append(clase_normalizada)

    # -------------------------------------------------------------
    # LÓGICA DE ESTABILIDAD Y ANUNCIO DE VOZ
    # -------------------------------------------------------------
    if len(elementos_validos) > 0:
        elemento_actual = elementos_validos[0]

        if elemento_actual == ultimo_detectado:
            contador_estabilidad += 1
        else:
            contador_estabilidad = 1
            ultimo_detectado = elemento_actual

        if contador_estabilidad >= FRAMES_REQUERIDOS and elemento_actual != ultimo_anunciado:
            frase_voz, valor_euro = CATALOGO_DINERO[elemento_actual]
            print(f"🔊 ANUNCIO: {frase_voz} ({valor_euro:.2f} €)")
            hablar_en_hilo(frase_voz)
            
            ultimo_anunciado = elemento_actual

    else:
        # Si no hay objeto en pantalla, reseteamos el estado
        contador_estabilidad = 0
        ultimo_detectado = None
        ultimo_anunciado = None

    # -------------------------------------------------------------
    # INTERFAZ VISUAL EN PANTALLA
    # -------------------------------------------------------------
    annotated_frame = results[0].plot()

    progreso = min(1.0, contador_estabilidad / FRAMES_REQUERIDOS)
    ancho_barra = int(300 * progreso)
    
    # Fondo del panel
    cv2.rectangle(annotated_frame, (20, 20), (320, 50), (40, 40, 40), -1)
    color_barra = (0, 255, 0) if contador_estabilidad >= FRAMES_REQUERIDOS else (0, 255, 255)
    
    # Barra de progreso
    cv2.rectangle(annotated_frame, (20, 20), (20 + ancho_barra, 50), color_barra, -1)
    
    # Texto en pantalla
    texto_pantalla = f"Estabilidad: {int(progreso*100)}%"
    if ultimo_detectado and ultimo_detectado in CATALOGO_DINERO:
        _, valor = CATALOGO_DINERO[ultimo_detectado]
        texto_pantalla += f" | {valor:.2f} EUR"
        
    cv2.putText(annotated_frame, texto_pantalla, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    cv2.imshow("Cajero Inteligente - Billetes & Monedas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()