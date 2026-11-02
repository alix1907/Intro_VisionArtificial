import os
import urllib.request
import cv2

# Descargar/verificar el archivo XML localmente para evitar errores de ruta en Windows
archivo_xml = "haarcascade_frontalface_default.xml"
if not os.path.exists(archivo_xml):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, archivo_xml)

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

detector = cv2.CascadeClassifier(archivo_xml)

if detector.empty():
    print("No se pudo cargar el clasificador Haar Cascade.")
    exit()

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo leer el fotograma de la cámara.")
        break

    # Convertir a escala de grises para la detección
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    rostros = detector.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Aplicar desenfoque (blur) a la región de cada rostro
    for (x, y, w, h) in rostros:
        # Recortar la Región de Interés (ROI) correspondiente al rostro
        rostro = frame[y:y+h, x:x+w]
        
        # Aplicar el filtro Gaussiano (los valores de kernel (99, 99) deben ser números impares)
        rostro_blur = cv2.GaussianBlur(rostro, (99, 99), 30)
        
        # Reemplazar la región original en el cuadro por el rostro desenfocado
        frame[y:y+h, x:x+w] = rostro_blur

    # Mostrar la ventana con el resultado
    cv2.imshow("Rostros desenfocados - Q para salir", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()
