from ovmsclient.tfs_compat.responses import PredictResponse, ModelMetadataResponse, ModelStatusResponse

class HttpPredictResponse(PredictResponse):
    pass

class HttpModelMetadataResponse(ModelMetadataResponse):
    
    def to_dict(self):
        pass

class HttpModelStatusResponse(ModelStatusResponse):
    
    def to_dict(self):
        pass

class HttpConfigStatusResponse:

    def to_dict(self):
        '''
        Return status in dictionary format:

        .. code-block::

            {
                ...
                <model_name>: {
                    ...
                    <version_number>: {
                        "state": <model_version_state>,
                        "error_code": <error_code>,
                        "error_message": <error_message>
                    },
                    ...
                },
                ...
            }

        '''
        pass
