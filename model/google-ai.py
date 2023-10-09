from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import base64
import time

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "europe-west2-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}

    # Initialize client that will be used to create and send requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())

    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    start = time.time()
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    end = time.time()

    print(f'\n prediction recieved in {round(end - start, 2)}s, writing to file...')

    # The predictions are a google.protobuf.Value representation of the model's predictions.
    image = base64.b64decode(response.predictions[0])

    # or, more concisely using with statement
    with open("imageToSave.png", "wb") as fh:
        fh.write(image)


predict_custom_trained_model_sample(
    project="587463540840",
    endpoint_id="473414522548256768",
    location="europe-west2",
    instances={ "prompt": "man playing guitar, sketch, high quality" }
)
