 Sample Project

Setup Elastic Search,  Kibana, Flask APP
--------------------------------------------------------

step 1:  docker-compose.yml file
step 2:  Run the above file to steup the containers
         docker-compose up -d   


Stop the containers

docker-compose down


kibana UI URL : http://localhost:5601/app

Elastic Search URl : http://localhost:9200

Run the application Container
-----------------------------------------------------------

Local Run
-----------------------------------

pip install -r requirements.txt

docker build -t elk_proj_flaskapp:latest .



Swagger Json
-----------------------------------------------------------

http://192.168.0.100:9097/swagger.json