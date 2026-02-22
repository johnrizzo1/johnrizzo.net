+++
draft = false
authors = ["John Rizzo"]
title = "A simple s3 client"
date = "2026-02-22"
tags = [
  "tools"
]
categories = [
  "tools"
]
series = [ ]
+++

Ever wish there were a simple, no-nonsense graphical client for S3? I've been frustrated by how clunky or overcomplicated most tools are — CyberDuck included — so I decided to build my own.

Simple S3 is exactly what the name suggests: a lightweight, single-purpose tool for managing files in S3 without the extra bloat or friction. It's built with Tauri and Rust, so it's fast and native on every platform.

![Simple S3](/images/simples3_screenshot.png)
Here's what it can do today:

- Dual-pane file browser — local files on the left, S3 on the right, just like a classic file manager
- Multiple S3 endpoints — connect to AWS S3, MinIO, Backblaze B2, Cloudflare R2, or any S3-compatible service and switch between them with a dropdown
- Secure credential storage — access keys are stored in your OS keychain (macOS Keychain, Windows Credential Manager, or Linux Secret Service), not in a config file
- Transfer queue — upload and download with pause, resume, and cancel
- Multipart transfers — large files over 100 MB are automatically split into parts
- Dark mode — follows your system theme or can be set manually
- Keyboard shortcuts — F5 to refresh, Delete to delete, Ctrl+U to upload, Ctrl+D to download
- Offline detection — transfers pause automatically when you lose connectivity and resume when it comes back
- Nix flake — NixOS users can run it directly with nix run or add it to their system packages

It's still in alpha, so expect some rough edges — but if you give it a try, I'd love your feedback (and bug reports!).

Check it out on GitHub: [Simple S3](https://github.com/johnrizzo1/simples3)