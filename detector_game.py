import os
import cv2
from ultralytics import YOLO

# 1. Cargamos el modelo "Nano" (el más rápido y ligero para webcam)
# La primera vez que se ejecute, se descargará automáticamente (unos 6 MB)
#model = YOLO("yolov8n.pt")
# 1. Localizamos la ruta de tu modelo recién entrenado
script_dir = os.path.dirname(os.path.abspath(__file__))

# NOTA: Revisa el nombre exacto de la carpeta dentro de 'runs/detect/'. 
# Si hiciste varias pruebas, puede llamarse 'mi_yolo_euro', 'mi_yolo_euro-2', etc.
modelo_path = os.path.join(script_dir, "runs", "detect", "mi_yolo_euro-5", "weights", "best.pt")

# Cargamos  el modelo entrenado
model = YOLO(modelo_path)

# 2. Iniciamos la cámara web
cap = cv2.VideoCapture(0)

print("¡Cámara iniciada! Presiona la tecla 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. ¡LA MAGIA DE YOLO!
    # Pasamos el fotograma directamente. conf=0.5 ignora detecciones con menos de 50% de certeza
    results = model(frame, conf=0.5)

    # 4. Dibujamos las cajas y etiquetas en la imagen original
    annotated_frame = results[0].plot()

    # En lugar de solo hacer results[0].plot(), inspeccionamos las cajas detectadas:
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obtenemos el nombre de la clase (ej. 'cell phone', 'person')
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # Obtenemos la confianza (de 0.0 a 1.0)
            confidence = float(box.conf[0])
            
            # ¡Imprimimos solo si detecta un teléfono o una persona!
            print(f"Objeto: {class_name} | Confianza: {confidence:.2f}")

    # 5. Mostramos el resultado
    cv2.imshow("Mi Primer Detector YOLOv8", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()