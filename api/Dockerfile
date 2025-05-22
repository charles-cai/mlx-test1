FROM python:3.9-slim
WORKDIR /app

# 1) Copy model artifact
COPY ../model/saved_model.pth /app/model/saved_model.pth
# 2) Copy API code
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
