import os
import urllib.request
import cv2

# Descargar el archivo XML oficial directamente al proyecto si no existe
archivo_xml = "haarcascade_frontalface_default.xml"
if not os.path.exists(archivo_xml):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    print("Descargando archivo de clasificador Haar Cascade...")
    urllib.request.urlretrieve(url, archivo_xml)

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Cargar el clasificador desde el archivo local
detector_rostros = cv2.CascadeClassifier(archivo_xml)

if detector_rostros.empty():
    print("No se pudo cargar el clasificador Haar Cascade.")
    exit()

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    # Convertir a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    rostros = detector_rostros.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Dibujar cuadro verde y etiqueta en cada rostro detectado
    for (x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Rostro",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Detector de rostros - presiona Q para salir", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()
