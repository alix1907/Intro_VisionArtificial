import cv2
from ultralytics import YOLO

# Cargar el modelo ligero preentrenado YOLOv8 Nano
# (se descargará automáticamente en la primera ejecución)
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

    # Realizar la inferencia con el modelo YOLO en el fotograma actual
    resultados = modelo(frame, verbose=False)

    # Renderizar las cajas delimitadoras y etiquetas devueltas por el modelo
    frame_anotado = resultados[0].plot()

    cv2.imshow("Deteccion de objetos YOLOv8 - presiona Q para salir", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()