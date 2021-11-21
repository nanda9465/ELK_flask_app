import requests
from elasticsearch import Elasticsearch
import settings
from api.utils.log_utils import create_logger_module

ELASTIC_LOGGER = create_logger_module("ElasticSearch")

def search(es_object, index_name, search):
    res = es_object.search(index=index_name, body=search)
    ELASTIC_LOGGER.info(res)


def create_index(es_object, index_name):
    created = False
    # index settings
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                    "type": {
                        "type": "text"
                    },
                    "product": {
                        "type": "text"
                    },
                    "usage": {
                        "type": "text"
                    },
                    "price": {
                        "type": "integer"
                    },
                    "currency": {
                        "type": "text"
                    },
                    "ip_addr": {
                        "type": "ip"
                    },
                    "location": {
                        "type": "geo_point"
                    }
            }
        }
    }

    try:
        if not es_object.indices.exists(index_name):
            # Ignore 400 means to ignore "Index Already Exist" error.
            es_object.indices.create(index=index_name, ignore=400, body=settings)
            ELASTIC_LOGGER.info('Created Index')
        created = True
    except Exception as ex:
        ELASTIC_LOGGER.error(str(ex))
    finally:
        return created


def store_record(elastic_object, index_name, record):
    is_stored = True
    try:
        outcome = elastic_object.index(index=index_name, doc_type='_doc', body=record)
        ELASTIC_LOGGER.info(outcome)
    except Exception as ex:
        ELASTIC_LOGGER.error('Error in indexing data')
        ELASTIC_LOGGER.error(str(ex))
        is_stored = False
    finally:
        return is_stored


def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': settings.ES_HOST, 'port': settings.ES_PORT}])
    if _es.ping():
        ELASTIC_LOGGER.info('Connection Successful')
    else:
        ELASTIC_LOGGER.error('Error Connecting Elastic Search')
    return _es
