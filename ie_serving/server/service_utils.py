

def check_availability_of_requested_model(models, model_name,
                                          requested_version):
    version = 0
    valid_model_spec = False
    requested_version = int(requested_version)
    if model_name in models:
        if requested_version == 0:
            version = models[model_name].default_version
            valid_model_spec = True
        elif requested_version in models[model_name].versions:
            version = requested_version
            valid_model_spec = True
    return valid_model_spec, version
