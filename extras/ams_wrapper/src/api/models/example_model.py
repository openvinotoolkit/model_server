from api.models.model import Model

class ExampleModel(Model):
   def postprocess_inference_output(self, inference_output: dict) -> str:
       #TODO: Implement postprocessing for example model
       return