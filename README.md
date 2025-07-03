# 🗑️ BinWatch – Smart Waste Container Monitoring

A full-stack web platform that detects whether public trash containers are **full or empty using only a photo**. Built for the smart-city space, the system uses lightweight python code to bring real-time visibility into waste bin usage.

https://trashia-production.up.railway.app/index.html

## 🔧 Tech Stack

**Frontend:**
- HTML, CSS, JavaScript
- Node.js (for SSR or tooling)

**Backend:**
- Python Flask (REST API)
- PostgreSQL (hosted on Railway)

**Deployment:**
- Railway (Backend & DB)
- Web-hosted frontend (static)

---

## 🚀 Features

- 📸 Upload a photo of a trash container
- 🤖 Model predicts: **Full / Empty**
- 🌐 Web-based interface, mobile-friendly
- 📊 Real-time detection using custom-trained model
- 🔌 Flask API for prediction inference
- 🗂️ Modular BackEnd and FrontEnd folders

---

## 🧠 Model

- The model is currently fine-tuned to a dataset from **Efrei Paris**.
- Detection is optimized for containers in public urban settings.
- Plans for broader bin/generalization detection in progress.

---

## 📁 Project Structure

```plaintext
/
├── BackEnd/         # Flask API and Javascript logic
├── FrontEnd/        # Static website or Node.js frontend
├── nixpacks.toml    # Deployment config (Railway)
