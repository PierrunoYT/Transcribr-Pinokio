module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git pull"
        ],
        on: [{
          event: "/error:/i",
          break: true
        }]
      }
    },
    // yt-dlp must stay current to keep working with YouTube, so always upgrade
    // it (and its pinned JS solver + runtime) instead of accepting what's installed.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt --upgrade-package yt-dlp --upgrade-package yt-dlp-ejs --upgrade-package deno"
        ],
        on: [{
          event: "/error:/i",
          break: true
        }]
      }
    },
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\""
        ],
        on: [{
          event: "/error:/i",
          break: true
        }]
      }
    },
    {
      method: "fs.write",
      params: {
        path: "env/transcribr.installed",
        text: "ok"
      }
    }
  ]
}
