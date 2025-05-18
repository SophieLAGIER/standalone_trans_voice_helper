import torch
import torchaudio
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tkinter as tk
from tkinter import ttk
from threading import Thread
import speech_recognition as sr
from model import ECAPA_gender
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time
import librosa

# Constantes
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.2
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
THRESHOLD = 0.01
SILENCE_DURATION = 1.0
MAX_RECORD_DURATION = 10.0

# Modèle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval().to(DEVICE)

def rms(block):
    return np.sqrt(np.mean(block**2))

def predict_gender(filepath):
    predicted_class, probabilities = model.predict(filepath, device=DEVICE)
    gender = "Female" if predicted_class == 1 else "Male"
    confidence = round(probabilities[predicted_class] * 100, 2)
    femininity_score = confidence if gender == "Female" else 100 - confidence
    return gender, confidence, femininity_score

def speech_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language='fr-FR')
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return "Service de reconnaissance indisponible"

class VoiceGenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur de genre et pitch vocal")

        # Graphe de pitch temps réel
        self.pitch_data = deque(maxlen=int(15 / BLOCK_DURATION))
        self.pitch_times = deque(maxlen=int(15 / BLOCK_DURATION))
        self.start_time = time.time()

        self.pitch_fig, self.pitch_ax = plt.subplots(figsize=(5, 2))
        self.pitch_line, = self.pitch_ax.plot([], [], marker='.')
        self.pitch_ax.set_ylim(50, 500)
        self.pitch_ax.set_xlim(0, 15)
        self.pitch_ax.set_ylabel("Pitch (Hz)")
        self.pitch_ax.set_title("Fréquence vocale - 15s")
        self.pitch_canvas = FigureCanvasTkAgg(self.pitch_fig, master=root)
        self.pitch_canvas.get_tk_widget().pack()

        self.label_status = tk.Label(root, text="En attente de parole...", font=("Arial", 14))
        self.label_status.pack(pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)

        self.text_label = tk.Label(root, text="", font=("Arial", 14), wraplength=500)
        self.text_label.pack(pady=10)

        # Graphe de féminité
        self.last_scores = []
        self.figure, self.ax = plt.subplots(figsize=(5, 2))
        self.line, = self.ax.plot([], [], marker='o')
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 9)
        self.ax.set_ylabel("Féminité (%)")
        self.ax.set_title("Dernières prédictions")
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

        # État
        self.recording = []
        self.recording_active = False
        self.recording_duration = 0
        self.silence_counter = 0

        self.thread = Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    def update_ui(self, level, status=None, result=None, text=None):
        self.progress['value'] = min(level * 1000, 100)
        if status:
            self.label_status.config(text=status)
        if result is not None:
            self.result_label.config(text=result)
        if text is not None:
            self.text_label.config(text=text)

    def update_graph(self, femininity_score):
        self.last_scores.append(femininity_score)
        self.last_scores = self.last_scores[-10:]
        self.line.set_ydata(self.last_scores)
        self.line.set_xdata(range(len(self.last_scores)))
        self.ax.set_xlim(0, max(9, len(self.last_scores)))
        self.canvas.draw()

    def update_pitch_graph(self):
        now = time.time()
        times = [now - t for t in self.pitch_times]
        self.pitch_line.set_xdata(times)
        self.pitch_line.set_ydata(self.pitch_data)
        self.pitch_ax.set_xlim(0, 15)
        self.pitch_canvas.draw()

    def get_pitch(self, audio_block):
        try:
            pitch = librosa.yin(audio_block, fmin=50, fmax=500, sr=SAMPLE_RATE)
            return float(np.nanmean(pitch))
        except:
            return None

    def listen_loop(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCK_SIZE) as stream:
            while True:
                block, _ = stream.read(BLOCK_SIZE)
                audio_block = block[:, 0]
                level = rms(audio_block)

                pitch = self.get_pitch(audio_block)
                if pitch and 50 < pitch < 500:
                    self.pitch_data.appendleft(pitch)  
                    self.pitch_times.appendleft(time.time())  
                    self.root.after(0, self.update_pitch_graph)

                self.root.after(0, self.update_ui, level)

                if level > THRESHOLD:
                    self.recording.append(audio_block.copy())
                    self.silence_counter = 0
                    self.recording_duration += BLOCK_DURATION
                    if not self.recording_active:
                        self.recording_active = True
                        self.root.after(0, self.update_ui, level, "Enregistrement...")

                    if self.recording_duration >= MAX_RECORD_DURATION:
                        force_process = True
                        reason = "Durée max atteinte"
                    else:
                        force_process = False
                        reason = ""
                elif self.recording_active:
                    self.silence_counter += BLOCK_DURATION
                    if self.silence_counter >= SILENCE_DURATION:
                        force_process = True
                        reason = "Silence détecté"
                    else:
                        force_process = False
                        reason = ""
                else:
                    force_process = False
                    reason = ""

                if force_process and self.recording_active:
                    self.root.after(0, self.update_ui, level, f"Traitement... ({reason})")
                    audio = np.concatenate(self.recording)
                    filename = "live_input.wav"
                    write(filename, SAMPLE_RATE, (audio * 32767).astype(np.int16))

                    transcription = speech_to_text(filename)

                    if transcription:
                        gender, confidence, femininity_score = predict_gender(filename)
                        result = f"👤 Genre : {gender} ({confidence} %)"
                        self.root.after(0, self.update_ui, 0, "En attente de parole...", result, transcription)
                        self.root.after(0, self.update_graph, femininity_score)

                        now = datetime.now().strftime("[%H:%M:%S]")
                        with open("historique.txt", "a", encoding="utf-8") as f:
                            f.write(f"{now} {transcription} | Genre : {gender} ({confidence} %)\n")
                    else:
                        self.root.after(0, self.update_ui, 0, "Aucune parole reconnue.", "—", "")

                    self.recording = []
                    self.recording_active = False
                    self.recording_duration = 0
                    self.silence_counter = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceGenderApp(root)
    root.mainloop()
