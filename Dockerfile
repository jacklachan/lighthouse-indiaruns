# Lighthouse — self-contained sandbox image (also used by the HuggingFace Docker Space).
# Runs the Streamlit demo on CPU. For the full 100K rank, use rank.py (see README).
FROM python:3.10-slim

WORKDIR /app

# CPU-only torch first (kept out of the CUDA default to keep the image small)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces expects the app on 7860
ENV PORT=7860
EXPOSE 7860
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
