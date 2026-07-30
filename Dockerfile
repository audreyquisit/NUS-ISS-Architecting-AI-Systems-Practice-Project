#Use an official python image
FROM python:3.9-slim

#Set work directory
WORKDIR /app

#Copy app files
COPY . .

#Install Flask
RUN pip install Flask

#Run the app
CMD ["python", "app.py"]