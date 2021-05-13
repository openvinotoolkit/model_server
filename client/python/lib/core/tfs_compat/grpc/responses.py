from core.tfs_compat.responses import PredictResponse, ModelMetadataResponse, ModelStatusResponse

class GrpcPredictResponse(PredictResponse):
    pass

class GrpcModelMetadataResponse(ModelMetadataResponse):
    
    def to_dict(self):
        raise NotImplementedError

class GrpcModelStatusResponse(ModelStatusResponse):
    
    def to_dict(self):
        raise NotImplementedError
