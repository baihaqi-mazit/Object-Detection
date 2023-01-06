# Installation: pip install octomizer-sdk --extra-index-url https://octo.jfrog.io/artifactory/api/pypi/pypi-local/simple

from octomizer import client, workflow
from octomizer.models import onnx_model

MY_ACCESS_TOKEN = 'ZaTlCC5wXpivnRxh7FWvEA=='        # 'Test' token

# Pass your API token below:
client = client.OctomizerClient(access_token=MY_ACCESS_TOKEN)

# Specify model file and input layer parameters.
model_file = "weights/best.onnx"

# Upload the model to Octomizer.
model = onnx_model.ONNXModel(client, name=model_file, model=model_file)

# Octomize it. By default, the resulting package will be a Python wheel.
wrkflow = model.get_uploaded_model_variant().octomize(platform="broadwell")
wrkflow.wait()
# Save the resulting Python wheel to the current directory.
wrkflow.save_package(".")