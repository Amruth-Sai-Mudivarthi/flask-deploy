# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install the ZBar library using apt-get (the system package manager)
RUN apt-get update && apt-get install -y libzbar0

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Your start command will be run by Render, so you don't need a CMD line here.
