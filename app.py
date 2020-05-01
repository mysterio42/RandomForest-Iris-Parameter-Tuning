from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from utils.model import load_model

app = Flask(__name__)
api = Api(app)
model = load_model()


class CreditScoring(Resource):

    def post(self):
        posted_data = request.get_json()

        assert 'sepalLength' in posted_data
        assert 'sepalWidth' in posted_data
        assert 'petalLength' in posted_data
        assert 'petalWidth' in posted_data
        pred = model.predict([
            list(posted_data.values())
        ])[0]

        return jsonify({'prediction': {'class': str(pred)}})


api.add_resource(CreditScoring, '/classify')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
