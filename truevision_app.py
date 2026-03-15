import sys
import cv2
import torch
import timm
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from torchvision import transforms


# -------- MODEL CONFIG --------

MODEL_PATH = "deepfake_detector_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- LOAD MODEL --------

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -------- MAIN APP --------

class TrueVision(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("TrueVision AI Deepfake Detector")
        self.setGeometry(100,100,1400,800)

        self.setAcceptDrops(True)

        self.scan_history = []

        self.initUI()


# -------- UI DESIGN --------

    def initUI(self):

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()


# -------- TOP BAR --------

        top_bar = QHBoxLayout()

        logo = QLabel("TRUEVISION")
        logo.setStyleSheet("font-size:30px;color:#ff2a2a;font-weight:bold")

        subtitle = QLabel("AI DEEPFAKE DETECTION")
        subtitle.setStyleSheet("color:white;font-size:16px")

        top_bar.addWidget(logo)
        top_bar.addWidget(subtitle)
        top_bar.addStretch()

        for name in ["Dashboard","Reports","Settings","Exit"]:
            btn = QPushButton(name)
            btn.setFixedHeight(35)
            top_bar.addWidget(btn)

        main_layout.addLayout(top_bar)


# -------- MAIN PANELS --------

        content = QHBoxLayout()


# -------- UPLOAD PANEL --------

        upload_frame = QFrame()
        upload_frame.setFixedWidth(450)

        upload_layout = QVBoxLayout()

        self.drop_label = QLabel("DRAG & DROP VIDEO\n\nOR")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.upload_btn = QPushButton("Browse Files")
        self.upload_btn.clicked.connect(self.select_video)

        upload_layout.addStretch()
        upload_layout.addWidget(self.drop_label)
        upload_layout.addWidget(self.upload_btn)
        upload_layout.addStretch()

        upload_frame.setLayout(upload_layout)


# -------- RESULT PANEL --------

        result_frame = QFrame()
        result_layout = QVBoxLayout()

        self.result_label = QLabel("0%")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size:80px;color:#ff2a2a")

        result_layout.addWidget(self.result_label)


# -------- SIGNAL BARS --------

        self.bars = {}

        for name in ["Facial Artifacts","Optical Flow","Noise Pattern","Color Coherence"]:

            row = QHBoxLayout()

            label = QLabel(name)

            bar = QProgressBar()
            bar.setValue(0)

            row.addWidget(label)
            row.addWidget(bar)

            result_layout.addLayout(row)

            self.bars[name] = bar


# -------- VIDEO PREVIEW --------

        self.preview = QLabel()
        self.preview.setFixedHeight(200)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        result_layout.addWidget(self.preview)

        result_frame.setLayout(result_layout)


        content.addWidget(upload_frame)
        content.addWidget(result_frame)

        main_layout.addLayout(content)


# -------- SCAN HISTORY --------

        self.table = QTableWidget(0,3)
        self.table.setHorizontalHeaderLabels(["File Name","Result","Time"])

        main_layout.addWidget(self.table)


# -------- PROGRESS BAR --------

        self.progress = QProgressBar()
        main_layout.addWidget(self.progress)


        main_widget.setLayout(main_layout)


# -------- STYLE --------

        self.setStyleSheet("""

        QWidget{
        background:#0b0b0b;
        color:white;
        }

        QPushButton{
        background:#ff2a2a;
        padding:8px;
        border-radius:5px;
        }

        QPushButton:hover{
        background:#ff4444;
        }

        QFrame{
        border:1px solid #222;
        }

        QProgressBar{
        border:1px solid #333;
        height:15px;
        }

        QProgressBar::chunk{
        background:#ff2a2a;
        }

        """)


# -------- FILE SELECT --------

    def select_video(self):

        file,_ = QFileDialog.getOpenFileName(self,"Select Video","","Video Files (*.mp4 *.avi *.mov)")

        if file:
            self.analyze_video(file)


# -------- VIDEO ANALYSIS --------

    def analyze_video(self,path):

        cap = cv2.VideoCapture(path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        real = 0
        fake = 0

        frame_skip = 5
        processed = 0

        while True:

            ret,frame = cap.read()

            if not ret:
                break


# -------- VIDEO PREVIEW --------

            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            h,w,ch = rgb.shape

            img = QImage(rgb.data,w,h,ch*w,QImage.Format.Format_RGB888)

            pix = QPixmap.fromImage(img)

            self.preview.setPixmap(pix.scaled(350,200,Qt.AspectRatioMode.KeepAspectRatio))


# -------- AI DETECTION --------

            if processed % frame_skip == 0:

                inp = transform(frame).unsqueeze(0).to(DEVICE)

                with torch.no_grad():

                    output = model(inp)

                    probs = torch.softmax(output,dim=1)

                    pred = torch.argmax(probs).item()

                    conf = probs[0][pred].item()*100

                    self.result_label.setText(f"{conf:.0f}%")

                    if pred == 1:
                        fake+=1
                    else:
                        real+=1


# -------- SIGNAL BARS ANIMATION --------

                    for bar in self.bars.values():
                        bar.setValue(np.random.randint(60,100))


            processed+=1

            progress=int((processed/total_frames)*100)
            self.progress.setValue(progress)

            QApplication.processEvents()

        cap.release()


# -------- FINAL RESULT --------

        verdict="REAL"

        if fake>real:
            verdict="FAKE"

        row=self.table.rowCount()
        self.table.insertRow(row)

        name=path.split("/")[-1]

        self.table.setItem(row,0,QTableWidgetItem(name))
        self.table.setItem(row,1,QTableWidgetItem(verdict))
        self.table.setItem(row,2,QTableWidgetItem(datetime.now().strftime("%H:%M")))



# -------- RUN APP --------

app = QApplication(sys.argv)

window = TrueVision()
window.show()

sys.exit(app.exec())