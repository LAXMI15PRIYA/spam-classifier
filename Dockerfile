# Start with an official Python image.
# This gives our Docker container Python 3.13.
FROM python:3.13-slim


# Create /app inside the container
# and make it our working folder.
#
# Future commands will run from /app.
WORKDIR /app


# Copy requirements.txt from our computer
# into the Docker container.
COPY requirements.txt .


# Install all Python packages
# needed by our project.
#
# --no-cache-dir keeps the Docker image smaller.
RUN pip install --no-cache-dir -r requirements.txt


# Copy our project files into the container.
#
# This includes:
# src/
# models/
# and other required files.
COPY . .


# Tell Docker that our FastAPI application
# uses port 8000.
EXPOSE 8000


# Start FastAPI when the container starts.
#
# src.app:app means:
# src/app.py
# and the FastAPI object named app.
#
# --host 0.0.0.0 is IMPORTANT in Docker.
# It allows requests from outside the container.
#
# We do NOT use --reload here because this
# container is meant to run the application,
# not continuously watch development files.
CMD ["sh", "-c", "python -m uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]