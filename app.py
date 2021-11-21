from flask import Flask
from flask_restx import Api
from api.v1.tracking_api import TRACKING_NS as trackingService
import settings

app = Flask(__name__)
API = Api(app, version='1.0', title='Tracking API')
API.add_namespace(trackingService, path='/api')


if __name__ == "__main__":
    app.run(host=settings.FLASK_HOST, port=settings.FLASK_PORT, threaded=True)
