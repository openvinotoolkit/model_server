from abc import ABC, abstractmethod

class PredictResponse(ABC):
    
    def __init__(self, raw_response):
        self.raw_response = raw_response

class ModelMetadataResponse(ABC):
    
    def __init__(self, raw_response):
        self.raw_response = raw_response
    
    @abstractmethod
    def to_dict(self):
        '''
        Return metadata in dictionary format:
        
        .. code-block::

            {
                ...
                <version_number>: {
                    "inputs": {
                        <input_name>: {
                            "shape": <input_shape>,
                            "dtype": <input_dtype>,
                        },
                            ...
                    },
                    "outputs":
                        <output_name>: {
                            "shape": <output_shape>,
                            "dtype": <output_dtype>,
                        },
                            ...
                    }
                },
                ...
            }

        '''

        pass

class ModelStatusResponse(ABC):
    
    def __init__(self, raw_response):
        self.raw_response = raw_response
    
    @abstractmethod
    def to_dict(self):
        '''
        Return status in dictionary format:

        .. code-block::

            {
                ...
                <version_number>: {
                    "state": <model_version_state>,
                    "error_code": <error_code>,
                    "error_message": <error_message>
                },
                ...
            }

        '''

        pass