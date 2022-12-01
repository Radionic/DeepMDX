import sys
import os
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QSize, Qt, QUrl, Slot
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, \
                              QLabel, QGridLayout, QWidget, QSlider

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep MDX")
        # Music player set-up
        # Documentation: https://doc.qt.io/qt-6/qml-qtmultimedia-mediaplayer.html
        self.player = QMediaPlayer()
        self.audioOutput = QAudioOutput()
        self.player.setAudioOutput(self.audioOutput)
        self.player.positionChanged.connect(self.positionChanged)
        self.player.durationChanged.connect(self.durationChanged)
        self.player.playbackStateChanged.connect(self.stateChanged)
        self.player.errorOccurred.connect(self._player_error)
        self.audioOutput.setVolume(0.5)
        # Initialisation
        self.app_state = 0 # No music
        self.aname = self.srcpath = self.outpath = ''
        # GUI
        layout = QGridLayout()
        # Course name title
        course_name = QLabel("AIST2010 Group Project\nDeep MDX")
        course_name.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(course_name, 0, 0, 1, 3)
        # Display loaded song
        self.song_name = QLabel("Load a song")
        self.song_name.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(self.song_name, 1, 0, 1, 3)
        # Upload button
        self.up_btn = QPushButton("Upload")
        self.up_btn.setCheckable(True)
        self.up_btn.clicked.connect(self.upload_music)
        layout.addWidget(self.up_btn, 2, 0)
        # Run MDX
        self.run_btn = QPushButton("Run")
        self.run_btn.setCheckable(True)
        self.run_btn.clicked.connect(self.run_mdx)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn, 2, 1)
        # Switch music
        self.switch_btn = QPushButton("Vocals")
        self.switch_btn.setCheckable(True)
        self.switch_btn.clicked.connect(self.switch_music)
        self.switch_btn.setEnabled(False)
        layout.addWidget(self.switch_btn, 2, 2)
        # Play button
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.play_music)
        self.play_btn.setEnabled(False)
        layout.addWidget(self.play_btn, 3, 0)
        # Pause button
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.pause_music)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn, 3, 1)
        # Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setCheckable(True)
        self.reset_btn.clicked.connect(self.reset_app)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn, 3, 2)
        # Volume slider
        slider_label = QLabel("Volume")
        slider_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(slider_label, 4, 0)
        slider1 = QSlider(Qt.Horizontal)
        slider1.setRange(0, 100)
        slider1.setSingleStep(1)
        slider1.setSliderPosition(50) # Default volume: 50
        slider1.setStyleSheet('''
            QSlider {
                border-radius: 10px;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #ccc;
            }
            QSlider::handle:horizontal{
                background: #ff2;
                width: 10px;
                height: 10px;
                margin: -4px 0;
                border-radius:4px;
            }
            QSlider::sub-page:horizontal{
                background:#666;
            }
        ''')
        slider1.valueChanged.connect(self.changevol)
        layout.addWidget(slider1, 4, 1, 1, 2)
        # Music slider
        slider_label = QLabel("Music")
        slider_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(slider_label, 5, 0)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(0, 1)
        self.slider2.setSingleStep(1)
        self.slider2.setEnabled(False)
        self.slider2.setStyleSheet('''
            QSlider {
                border-radius: 10px;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #ccc;
            }
            QSlider::handle:horizontal{
                background: #808;
                width: 10px;
                height: 10px;
                margin: -4px 0;
                border-radius:4px;
            }
            QSlider::sub-page:horizontal{
                background:#666;
            }
        ''')
        self.slider2.sliderPressed.connect(self.slider_phandler)
        self.slider2.sliderReleased.connect(self.slider_rhandler)
        self.slider2.sliderMoved.connect(self.movepos)
        layout.addWidget(self.slider2, 5, 1, 1, 2)
        
        self.setFixedSize(QSize(400, 250)) # Set window size
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    # Current music time
    @Slot()
    def positionChanged(self, position):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.slider2.setSliderPosition(position)

    # Music changed
    @Slot()
    def durationChanged(self, duration):
        self.slider2.setRange(0, duration)

    # Audio player state: "PlayingState", "PausedState", "StoppedState"
    @Slot()
    def stateChanged(self, state):
        pass

    # Error handling for audio player
    @Slot()
    def _player_error(self, error, error_string):
        print(error, error_string)

    # Upload music button on-click
    def upload_music(self):
        # Documentation: https://doc.qt.io/qt-6/qfiledialog.html
        fname = QFileDialog.getOpenFileName(self, 'Select an audio', os.getcwd(), "Audio files (*.wav *.mp3)")
        if fname[0] != '': # File selected
            self.srcpath = fname[0]
            self.aname = self.srcpath.rsplit('/', 1)[-1].rsplit('.', 1)[0] # Audio name
            self.song_name.setText(f"Song: {self.aname}")
            self.app_state = 1 # Music uploaded
            self.play_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.slider2.setEnabled(True)
            self.switch_btn.setEnabled(False)
            self.player.stop()
            self.pause_btn.setChecked(False)
            self.pause_btn.setEnabled(False)
            self.slider2.setSliderPosition(0)
            self.player.setSource(QUrl.fromLocalFile(self.srcpath))
        self.up_btn.setChecked(False)
    
    # Run MDX button on-click
    def run_mdx(self):
        fname = QFileDialog.getExistingDirectory(self, 'Choose output Directory', os.getcwd())
        self.outpath = fname
        if self.outpath != '' and self.srcpath != '' and self.app_state == 1: # File selected and directory selected
            self.play_btn.setChecked(False)
            self.pause_btn.setChecked(False)
            self.pause_btn.setEnabled(False)
            self.player.stop()
            self.slider2.setSliderPosition(0)
            os.system(f"python ./DeepMDX/pl_inference.py -i {self.srcpath} -P ./DeepMDX/checkpoints/best.ckpt -o {self.outpath}")
            self.app_state = 2 # MDX completed, Instrument
            self.player.setSource(QUrl.fromLocalFile(self.outpath + '/' + self.aname + '_Instruments.wav'))
            self.switch_btn.setEnabled(True)
            self.run_btn.setEnabled(False)
        self.run_btn.setChecked(False)

    # Switch music button on-click
    def switch_music(self):
        self.switch_btn.setChecked(False)
        self.player.stop()
        self.slider2.setSliderPosition(0)
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(False)
        if self.app_state == 2: # Inst to vocal
            self.switch_btn.setText("Original")
            self.player.setSource(QUrl.fromLocalFile(self.outpath + '/' + self.aname + '_Vocals.wav'))
            self.app_state = 3
        elif self.app_state == 3: # Vocal to original
            self.switch_btn.setText("Instruments")
            self.player.setSource(QUrl.fromLocalFile(self.srcpath))
            self.app_state = 4
        elif self.app_state == 4: # Original to inst
            self.switch_btn.setText("Vocals")
            self.player.setSource(QUrl.fromLocalFile(self.outpath + '/' + self.aname + '_Instruments.wav'))
            self.app_state = 2

    # Play music button on-click
    def play_music(self):
        self.player.play()
        self.play_btn.setChecked(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(True)
    
    # Pause music button on-click
    def pause_music(self):
        self.pause_btn.setChecked(True)
        self.play_btn.setChecked(False)
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()

    # Reset app button on-click
    def reset_app(self):
        self.reset_btn.setChecked(False)
        # Reset variables
        self.app_state = 0
        self.aname = self.srcpath = self.outpath = ''
        # Reset player
        self.player.stop()
        self.song_name.setText("Load a song")
        # Reset slider
        self.slider2.setSliderPosition(0)
        self.slider2.setEnabled(False)
        # Reset buttons
        self.run_btn.setEnabled(False)
        self.switch_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.play_btn.setChecked(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setChecked(False)
        self.reset_btn.setEnabled(False)
    
    # Volume-slider on-change
    def changevol(self, vol):
        self.audioOutput.setVolume(vol / 100)
    
    # Music slider on-move
    def movepos(self, position):
        self.player.setPosition(position)

    # Pause music when choosing music position
    def slider_phandler(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.pause_btn.setChecked(True)
            self.play_btn.setChecked(False)
    
    # Resume music when choosing music position
    def slider_rhandler(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self.player.play()
            self.pause_btn.setChecked(False)
            self.play_btn.setChecked(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()