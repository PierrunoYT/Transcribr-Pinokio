module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "notify",
      params: {
        html: "Installing Transcribr..."
      }
    },
    // Install transcription + download dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt"
        ],
      }
    },
    // NVIDIA GPU acceleration: faster-whisper (CTranslate2) needs cuBLAS + cuDNN
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ Installed! Whisper models download on first run. Works on GPU or CPU."
      }
    },
    {
      method: "script.start",
      params: {
        uri: "start.js"
      }
    }
  ]
}
