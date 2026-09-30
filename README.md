# Ollama Screen Assistant

A small, always-on-top desktop overlay that sends the primary-screen image and your task to a local Ollama vision model. The model can propose mouse and keyboard actions; each action requires your approval before it runs.

## Run

1. Install Ollama and start its local service.
2. Pull a vision model with tool-calling support, for example:

   ```sh
   ollama pull qwen3-vl:4b
   ```

3. Install dependencies and start the overlay:

   ```sh
   python3 -m pip install -r requirements.txt
   python3 app.py
   ```

The model field can be changed to another Ollama vision model that supports tool calls. The app uses Ollama at `127.0.0.1:11434`; screen images are not sent to a hosted service.

## Platform notes

Global screen capture and input are most reliable in an X11 desktop session. Wayland restricts these capabilities; this overlay detects Wayland and warns that capture or input may not work. On Linux, PyAutoGUI may also require system packages for X11 and Tk. The app only targets the primary monitor.

Every proposed click, typed string, key, shortcut, or scroll must be approved in the overlay. Keep the overlay visible while a task runs, and stop it at any time. The model is limited to those actions and does not have shell access.