# Lighthouse — self-contained sandbox image (HuggingFace Docker Space).
# Serves the FastAPI app + animated frontend on CPU. For the full 100K rank, use rank.py.
FROM python:3.10-slim

WORKDIR /app

# CPU-only torch first (from the PyTorch CPU wheel index).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# App/runtime deps: the FastAPI server plus the small encoder used at request time.
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
