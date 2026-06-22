FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Command to run the Streamlit application on port 7860
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
