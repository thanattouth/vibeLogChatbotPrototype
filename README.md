# 🛡 MilitarySOCAnalysisChatbot_Prototype

ระบบต้นแบบ (Prototype) สำหรับวิเคราะห์ Log ทางด้านความมั่นคงปลอดภัยทางไซเบอร์ โดยใช้ AI/LLM, LangChain, FAISS และ Gradio UI เหมาะสำหรับใช้งานในระบบ Military SOC (Security Operations Center) ที่ต้องการวิเคราะห์ log แบบ offline / air-gapped

---

## ✨ คุณสมบัติ

- วิเคราะห์ log ด้วย AI (LLM) แบบ offline ผ่าน [Ollama](https://ollama.com)
- สร้างเวกเตอร์จาก log และค้นหาด้วย FAISS (Vector Similarity Search)
- ใช้ HuggingFace Embedding + LangChain RetrievalQA
- รองรับ UI ผ่าน Gradio สำหรับการทดลองใช้งานง่าย
- Modular Code พร้อมต่อยอดเป็นระบบขนาดใหญ่

---

## 🧠 ตัวอย่างการใช้งาน

1. ผู้ใช้ต้องส่ง log เป็นไฟล์ .log | .txt | .json | .csv
2. จากนั้นผู้ใช้ถามคำถาม เช่น มีเหตุการณ์ผิดปกติอะไรเกิดขึ้นบ้าง
3. จากนั้น Chatbot จึงจะตอบ

---

## 🔧 การติดตั้ง

### 1. Clone โปรเจกต์

```bash
git clone https://github.com/thanattouth/vibe_log_chatbot_protorype
cd vibe_log_chatbot_protorype

# สร้าง Virtual Environment
python -m venv venv
source .venv/bin/activate   # Windows: venv\Scripts\activate

# ติดตั้ง Dependencies
pip install -r requirements.txt
```

การใช้งาน
1. ติดตั้งและโหลดโมเดลจาก Ollama

ติดตั้ง Ollama:
https://ollama.com

เปิด Ollama:
ollama serve

โหลดโมเดล phi3:mini:
ollama pull phi3:mini

2. รันโปรแกรม
python logbot_prototype.py
Gradio UI จะเปิดที่ http://0.0.0.0:7860 โดยอัตโนมัติ

MIT License © 2025 Thanattouth