from kivymd.app import MDApp
import cv2
import numpy as np
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        model, classes, colors, output_layers = self.load_yolo()
        if ret:
            height, width, channels = frame.shape  # Prendo le informazioni relativi al frame su altezza larghezza e canale
            blob, outputs = self.detect_objects(frame, model, output_layers)  # Funzione di detect
            boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)  # Prendo informazioni del box
            frame = self.draw_labels(boxes, confs, colors, class_ids, classes, frame)
            #converti il frame in texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # Visualizza l'immagine dalla texture
            self.texture = image_texture

    def load_yolo(self):
        net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg") #Carico i due file di Yolo nella mia rete neurale
        #classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()] #Carico le classi che Yolo riconosce (in coco.names) in classes

        output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]  # restituisce gli indici dei livelli di output della rete.
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        return net, classes, colors, output_layers


    def detect_objects(self,img, net, outputLayers):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False) #Prepara l'immagine per essere analizzata dalla rete
        net.setInput(blob)
        outputs = net.forward(outputLayers) #Restituisce le informazioni relative agli oggetti analizzati
        return blob, outputs


    def get_box_dimensions(self, outputs, height, width):  #Prendiamo le misure per il box relativo all'oggetto
        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores) #Prendiamo l'id massimo
                conf = scores[class_id]
                if conf > 0.3: #Se la confidenza supera 0.3 allora prendiamo le misure per il riquadro
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids


    def draw_labels(self, boxes, confs, colors, class_ids, classes, img): #Disegniamo il riquadro attorno agli oggetti
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4) #Per selezionare un solo riquadro visto che con Yolo ne potremo avere di più
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]).upper()
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), cv2.FONT_ITALIC, 1, color, 1)
        return img


class MainApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture = self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        # per chiudere la finestra
        self.capture.release()


if __name__ == '__main__':
    # run app
    MainApp().run()
