FROM bitnami/python:3.9.13

# Move in server folder
WORKDIR /bodegic

# Copy requirements.txt and install all dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files in server directory
COPY . .

# Run Bodegic.py
ENTRYPOINT [ "python",  "bodegic.py"]
CMD ["--help"]
