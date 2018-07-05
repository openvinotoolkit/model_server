

def check_if_model_name_and_version_is_valid(model_spec, available_models):
    if model_spec.name in list(available_models.keys()):
        if model_spec.version.value == 0:
            version = available_models[model_spec.name].default_version
            return True, model_spec.name, version
        elif int(model_spec.version.value) in available_models[model_spec.name].versions:
            version = available_models[model_spec.name].default_version
            return True, model_spec.name, version
    return False, model_spec.name, model_spec.version.value
