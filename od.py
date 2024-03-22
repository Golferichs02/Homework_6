import cv2
import numpy as np
import argparse as arg
from numpy.typing import NDArray

#Función para interactuar con el usuario 
def user_interaction()->arg.ArgumentParser:
    parser = arg.ArgumentParser(description='Object detection')
    parser.add_argument("-v",'--video_file', 
                        type=str, 
                        default='camera', 
                        help='Video file used for the object detection process')
    parser.add_argument('--frame_resize', 
                        type=int, 
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    args = parser.parse_args()
    if args.frame_resize > 90:
        parser.error("resize must be under 20%.")
    return args

def initialise_camera(args:arg.ArgumentParser)->cv2.VideoCapture:
    
    # Create a video capture object
    cap = cv2.VideoCapture(args.video_file)
    return cap 

#Función para Reescalar el frame de en base al porcentaje del frame original
def rescale_frame(frame:NDArray, percentage:np.intc=20)->NDArray:
    
    # Resize current frame
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

#Función para filtrar los colores de interes
def hsv_segmentation(frame:NDArray)->NDArray:
        # Convert the current frame from BGR to HSV
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Definir los rangos de color para el negro y el azul
        lower_black = np.array([0, 0, 10])
        upper_black = np.array([180, 255, 30])  # El rango de saturación y valor puede variar según tus necesidades
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        # Aplicar un umbral a la imagen HSV para obtener solo los píxeles que caen dentro de los rangos de color especificados
        mask_black = cv2.inRange(frame_HSV, lower_black, upper_black)
        mask_blue = cv2.inRange(frame_HSV, lower_blue, upper_blue)

        # Combinar las máscaras para obtener solo los píxeles que son negro o azul de forma binarizada
        mask_combined = cv2.bitwise_or(mask_black, mask_blue)
        return mask_combined

# Función para minimizar la región de busqueda de los pixeles
def minimize_box(matriz:NDArray,current:int)->NDArray:
    l = []
    for i in range(len(matriz)):
        if matriz[i] < current+10 and matriz[i] > current-10:
            l.append(matriz[i])
    return l

#Función que devuelve las coordenadas del objeto que se esta trackeando
def object_detection(mask:NDArray,y_current:int = 0,x_current:int=0, t:int = 1)->tuple[int,int,int]:
    check = mask.nonzero()
    
    if check[0].any() == False:
        t = 1
        y_current = -10
        x_current = -10

    if t == 1 & check[0].any():
        y = np.array(check[0])
        x = np.array(check[1])
        y_current = int(np.mean(y))
        x_current = int(np.mean(x))
        t = 2

    if check[0].any:
        y = np.array(check[0])
        x = np.array(check[1])
        y = minimize_box(y,y_current)
        x = minimize_box(x,x_current)
        y = np.mean(y)
        x = np.mean(x)
        if y > y_current-10 and y < y_current+10 and x> x_current-10 and x < x_current+10:
            y_current = int(y)
            x_current = int(x)
    

    return y_current, x_current, t

#Función que implementa el filtrado y el tracker asi como dibujar el rectangulo de detección en la imagen original
def segment_object(cap:cv2.VideoCapture, args:arg.ArgumentParser)->None:
    y = 0
    x = 0
    t = 1
    # Main loop
    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()

        # Check if the image was correctly captured
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        # Resize current frame
        frame = rescale_frame(frame, args.frame_resize)
        
        #Aplicar el filtro de HSV para obtener una matríz binarizado  
        mask = hsv_segmentation(frame) 

        #Obtener las coordenadas del objeto detectado
        y,x,t = object_detection(mask,y,x,t)

        #Dibujar en la imagen original el rectangulo respecto a las coordenadas obtenidas. 
        cv2.rectangle(frame,(x-10,y+10), (x+10,y-10),(255,255,255),2) 

        # Visualizar los videos del filtro y el original
        cv2.imshow("original_video", frame)
        cv2.imshow("Filtered_image", mask)

        #El programa se finaliza si se presiona la letra q
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break

def close_windows(cap:cv2.VideoCapture)->None:
    
    # Destruir ventanas
    cv2.destroyAllWindows()

    # Destruir el objeto de captura de video.
    cap.release()