#Use an official python image
FROM python:3.9-slim

#Set work directory
WORKDIR /app

# Copy requirements first
COPY frontend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy app files
COPY . .


#Run the app
CMD ["python", "app.py"]