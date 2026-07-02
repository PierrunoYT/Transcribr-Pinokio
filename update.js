module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git pull"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt"
        ]
      }
    },
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
        ]
      }
    }
  ]
}
