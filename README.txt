Importa i file nel tuo progetto Python.
Scarica il file weights di Yolov3-Tiny dal seguente link: https://pjreddie.com/darknet/yolo/
Importa il file nel progetto.
Nelle impostazioni del progetto importa le seguenti librerie: kivy, opencv-python, numpy, kivymd, cython

SE VUOI CAMBIARE IN YOLOV3
Scarica i file wrights e cfg dal sito: https://pjreddie.com/darknet/yolo/
Importali nel progetto *Elimina i precedenti di Yolov3-Tiny*
Modifica il contenuto di cv2.dnn.readNet (line 33) con i nomi dei file che hai appena scaricato (prima weights poi cfg)

Have fun!!
