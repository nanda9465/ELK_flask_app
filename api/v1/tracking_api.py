from http import HTTPStatus
import requests
import settings
from flask import request
from flask_restx import Resource, reqparse, inputs
from api.v1 import TRACKING_NS
from api.utils.log_utils import create_logger_module
from api.utils.elastic_search import connect_elasticsearch, create_index, store_record

SCHEMA_ARG_PARSER = reqparse.RequestParser()
API_LOGGER = create_logger_module("API Logger")
GEOLOCATION_URL = 'https://geolocation-db.com/json/'
 
@TRACKING_NS.route("/tracking")
class GetTracking(Resource):
    SCHEMA_ARG_PARSER.add_argument("type", type=str, required=True)
    SCHEMA_ARG_PARSER.add_argument('product', type=str, required=True)
    SCHEMA_ARG_PARSER.add_argument('usage', type=str, required=True)
    SCHEMA_ARG_PARSER.add_argument('price', type=inputs.positive, required=True)
    SCHEMA_ARG_PARSER.add_argument("currency", type=str, required=True)
    @staticmethod
    @TRACKING_NS.doc('GET Tracking details')
    @TRACKING_NS.response(200, 'Success')
    @TRACKING_NS.response(401, 'Unauthorized')
    @TRACKING_NS.response(500, 'Internal Server Error')
    @TRACKING_NS.response(400, 'Bad Request')
    @TRACKING_NS.expect(SCHEMA_ARG_PARSER, validate=True)
    def get():
        """GET Tracking details"""
        message_obj = SCHEMA_ARG_PARSER.parse_args()
        ip_address= request.remote_addr   # gives the IP address
        if ip_address.split('.')[0] in settings.PRIVATE_ADDRESS:
            # local testing - private IP address range dummy IP address
            request_url = GEOLOCATION_URL + settings.DUMMY_IP
        else:
            request_url = GEOLOCATION_URL + ip_address
        geo_location_response = requests.get(request_url).json()
        message_obj['ip_addr'] = ip_address
        message_obj['location'] = {'lat' : geo_location_response['latitude'],
                                    'lon' : geo_location_response['longitude']}
        es = connect_elasticsearch()
        if es is not None:
            if create_index(es, 'tracking'):
                out = store_record(es, 'tracking', message_obj)
                API_LOGGER.info('Data indexed successfully')
        API_LOGGER.info(message_obj)
        return message_obj, HTTPStatus.OK


@TRACKING_NS.route('/health')
class HealthResource(Resource):
    @staticmethod
    @TRACKING_NS.doc('Health API')
    @TRACKING_NS.response(200, 'Success')
    @TRACKING_NS.response(401, 'Unauthorized')
    @TRACKING_NS.response(400, 'Bad Request')
    def get():
        """Health API"""
        return {'Status': 'Success'}, HTTPStatus.OK
