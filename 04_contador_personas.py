import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 Nano
modelo = YOLO("yolov8n.pt")

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    # Realizar la detección
    resultados = modelo(frame, conf=0.5, verbose=False)
    
    # Generar el frame anotado con las detecciones de YOLO
    frame_anotado = resultados[0].plot()

    # Contar la cantidad de personas detectadas en la toma
    conteo_personas = 0
    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        nombre = modelo.names[clase_id]
        if nombre == "person":
            conteo_personas += 1

    # Imprimir el contador en la pantalla
    cv2.putText(
        frame_anotado,
        f"Personas: {conteo_personas}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Mostrar el video en vivo
    cv2.imshow("Contador de personas - Q para salir", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()