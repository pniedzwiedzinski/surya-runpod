""" Example handler file. """

import runpod

from ocr import ocr_text

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    paths = job_input.get('images')

    response = ocr_text(paths)

    return response


runpod.serverless.start({"handler": handler})
