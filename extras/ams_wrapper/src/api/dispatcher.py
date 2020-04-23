import falcon
from api.ovms_connector import OvmsConnector
from config import OVMS_PORT
def create_dispatcher(available_models: list):
    dispatch_map = {}
    for available_model in available_models:
        ovms_connector = OvmsConnector(OVMS_PORT, available_model['ovms_mapping'])
        model = available_model['class'](ovms_connector)
        dispatch_map[available_model['name']] = model

    dispatcher = falcon.API()

    for target_model, request_handler in dispatch_map.items():
        dispatcher.add_route(f"/{target_model}", request_handler)
    
    return dispatcher


