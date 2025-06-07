# 🎛️ Gesture-Based Volume Control (Linux)

Control your computer's system volume with simple hand gestures using your webcam. This project uses **MediaPipe** for real-time hand tracking, **OpenCV** for video capture, and Linux’s `amixer` utility to change system volume — no mouse or keyboard needed.

---

## 🚀 Features

- ✅ Real-time hand tracking using MediaPipe
- ✅ Adjust system volume using thumb–index finger distance
- ✅ Visual volume level overlay for feedback
- ✅ Fully offline and runs on CPU
- ✅ Lightweight, fast, and beginner-friendly
- ✅ Works on most Linux distributions

---

## 🖼️ Demo

https://github.com/your-username/gesture-volume-control-linux/assets/demo.gif  
*(Add your screen recording or gif demo here for better visibility)*

---

## 🛠️ Tech Stack

| Tool        | Purpose                       |
|-------------|-------------------------------|
| OpenCV      | Webcam access and visualization |
| MediaPipe   | Hand landmark detection       |
| NumPy       | Numeric operations            |
| amixer (ALSA) | Linux audio control          |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gesture-volume-control-linux.git
cd gesture-volume-control-linux
