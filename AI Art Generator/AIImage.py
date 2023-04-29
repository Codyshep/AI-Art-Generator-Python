import io
import os
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Our Host URL should not be prepended with "https" nor should it have a trailing slash.
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-kFdBueouATOdCoannfXQZzapjNzvbY0mwXAf6LRVZETl1tLC'

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-beta-v2-2-2", 
)

# Set up our initial generation parameters.
answers = stability_api.generate(
    prompt= input("What Would You Like To Generate"),
    seed=892226758,
    steps=30, 
    cfg_scale=8.0, 
    width=512,
    height=512,
    sampler=generation.SAMPLER_K_DPMPP_2M 
)

for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            global img
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(str(artifact.seed)+ ".png") # Save our generated images its seed number as the filename.