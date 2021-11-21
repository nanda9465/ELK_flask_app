import os

ES_HOST = os.getenv('ELASTICSEARCH_HOST', 'elasticsearch')
ES_PORT = os.getenv('ELASTICSEARCH_PORT', 9200)
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = os.getenv('FLASK_PORT', 9097)
PRIVATE_ADDRESS = ['172', '127', '192', '0']
DUMMY_IP = os.getenv('DUMMY_IP', '39.110.142.79')