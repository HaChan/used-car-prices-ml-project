FROM python:3.11-slim

# Create a working directory for the application
WORKDIR /app

# Copy the requirements file and install dependencies
COPY predict_requirements.txt .
RUN pip install --no-cache-dir -r predict_requirements.txt

COPY model.bin ./
COPY gunicorn_config.py app.py ./

RUN useradd -m deploy
RUN chown -R deploy:deploy /app
USER deploy
EXPOSE 8000

CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
