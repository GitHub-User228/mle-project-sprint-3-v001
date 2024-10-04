# Instructions on how to start the microservice

Each instruction is executed from the repository directory

## 1. FastAPI microservice via conda environment

Install Conda on the machine and initialize it before proceeding.

```python
# creating a conda environment
conda create -y --name venv1 python=3.10

# activating the conda environment
conda activate venv1

# exporting the environment variables for the .env
conda env config vars set $(cat services/.env | tr '\n' ' ')

# reactivating the conda environment
conda deactivate
conda activate venv1

# cd to the directory with the services
cd services/ml_service

# installing the dependencies
pip install -r requirements.txt

# starting the microservice via uvicorn
uvicorn app.flat_price_app_simple:app --reload --host 0.0.0.0 --port 8123
```

A different port is used to avoid conflicts with the future docker-based microservice

### Example curl query to the microservice

```bash
curl -X 'POST' \
  'http://localhost:8123/predict?user_id=1' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "has_elevator": true,
  "is_apartment": false,
  "building_type_int": 5,
  "rooms": 1,
  "floor": 6,
  "floors_total": 17,
  "flats_count": 198,
  "build_year": 1915,
  "ceiling_height": 2.7,
  "kitchen_area": 10,
  "living_area": 18,
  "total_area": 75,
  "latitude": 55.844349,
  "longitude": 37.349083
}'
```

This query should be valid and return the following response: `{"user_id":"1","prediction":10803021.673164235}`

If you are using the same terminal where the microservice is running, follow these steps:

```bash
# stopping the service (press this)
Ctrl + C

# deactivating the conda environment
conda deactivate
```

Do not delete this conda environment, since it will be used in the future step.


## 2. FastAPI microservice via Docker container

```bash
# assuming you are in the root directory of the repository, cd to the directory with the services
cd services

# building an image from Dockerfile
docker build -t ml_service_image -f Dockerfile_single_service .

# running the container from the created image
docker run --name ml_service_container --publish 4602:8081 --volume=./models:/fastapi_app/models --env-file .env ml_service_image
```

### Example curl query to the microservice

```bash
curl -X 'POST' \
  'http://localhost:4602/predict?user_id=1' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "has_elevator": true,
  "is_apartment": false,
  "building_type_int": 5,
  "rooms": 1,
  "floor": 6,
  "floors_total": 17,
  "flats_count": 198,
  "build_year": 1915,
  "ceiling_height": 2.7,
  "kitchen_area": 10,
  "living_area": 18,
  "total_area": 75,
  "latitude": 55.844349,
  "longitude": 37.349083
}'
```

This query should be valid and return the following response: `{"user_id":"1","prediction":10803021.673164235}`

### Stopping the docker container

It is necessary to stop the container before procceding to the next step since the ports used in this case are required for the next step and need to be free.

```bash
# Run this in another terminal
docker container stop ml_service_container
```

## 3. Docker compose for microservice and monitoring system

```bash
# assuming you are in the root directory of the repository, cd to the directory with the services
cd services

# building the image and starting the services via docker compose
docker compose up --build
```

### Example curl query to the microservice

```bash
curl -X 'POST' \
  'http://localhost:4602/predict?user_id=1' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "has_elevator": true,
  "is_apartment": false,
  "building_type_int": 5,
  "rooms": 1,
  "floor": 6,
  "floors_total": 17,
  "flats_count": 198,
  "build_year": 1915,
  "ceiling_height": 2.7,
  "kitchen_area": 10,
  "living_area": 18,
  "total_area": 75,
  "latitude": 55.844349,
  "longitude": 37.349083
}'
```

This query should be valid and return the following response unless you exceeded the maximum number of requests from a single IP or the global limit: `{"user_id":"1","prediction":10803021.673164235}`  

In case you exceeded the limit, you will see the response similar to one of the following: 
- `{"detail":"Too Many Requests from your IP. Retry after 14 seconds."}`
- `{"detail":"Too Many Overall Requests. Retry after 14 seconds."}`


## 4. Simulation of the load on the microservice

Script [test.py](services/ml_service/tests/test.py) simulates a load on the microservice and have several arguments which control the number of requests, the time interval between requests and whether to use a different IP address for each request. Check the script to see more info.

```bash
# If you are not yet in the conda environment created eariler, activate it
conda activate venv1

# Assuming you are in the root directory of the repository, cd to the directory with the microservice
cd services/ml_service

# Run the testing script to send 50 different valid requests with a 0.01 second delay from different IPs with 0 invalid requests
python3 tests/test.py --r 50 --d 0.01 --m True --i 0.0

# Run the testing script to send 10 different valid requests with a 0.01 second delay from the same IP with 30% chance of sending an invalid request
python3 tests/test.py --r 10 --d 0.01 --m False --i 0.3
```

Addresses of the services:
- microservice: [http://localhost:4602](http://localhost:4602)
- Prometheus: [http://localhost:3000](http://localhost:3000)
- Grafana: [http://localhost:9090](http://localhost:9090)