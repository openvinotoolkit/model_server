class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        # This method will be called once during initialization.
        # It expects keyword arguments. They will map node configuration from pbtxt including node options, node name etc.
        # Detailed spec to be provided. 
        ...
        return None

    def execute(self, inputs: list, kwargs: dict) -> list:
        # This method will be called for every request.
        # It expects a list of inputs (our custom python objects).
        # It also expects keyword arguments. They will be provided by the calculator to enable advanced processing and flow control.
        # Detailed spec to be provided. 
        #
        # It will returns list of outputs (also our custom python objects).
        ...
        return outputs

    def finalize(self, kwargs: dict):
        # This method will be called once during deinitialization. 
        # It expects keyword arguments. They will map node configuration from pbtxt including node options, node name etc.
        # Detailed spec to be provided. 
        ...
        return None