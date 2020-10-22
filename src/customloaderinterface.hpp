#ifndef CUSTOM_LOADER_INTERFACE_CLASS_H
#define CUSTOM_LOADER_INTERFACE_CLASS_H

#include <iostream>
#include <map>
#include <string>

namespace ovms {

/**
     * @brief This class is the custom loader interface base class.
     * Custom Loaders need to implement this interface and define the virtual functions to enable
     * OVMS load a model using a custom loader
     */
class CustomLoaderInterface {
public:
    /**
         * @brief Constructor
         */
    CustomLoaderInterface() {
        printf("Created custom loader object................\n");
    }
    /**
         * @brief Destructor
         */
    virtual ~CustomLoaderInterface() {
        printf("Deleted custom loader object................\n");
    }

    /**
         * @brief Initialize the custom loader
         *
         * @param loader config file defined under custom loader config in the config file
         *
         * @return status
         */
    virtual int loaderInit(char* loaderConfigFile) = 0;

    /**
         * @brief Load the model by the custom loader
         *
         * @param model name required to be loaded - defined under model config in the config file
         * @param base path where the required IR files are present
         * @param version of the model
         * @param loader config parameters json as string
         * @param char pointer to the model xml buffer
         * @param length of the model xml buffer
         * @param char pointer to the weights buffer
         * @param length of the weights buffer
         * @return status
         */
    virtual int loadModel(const char* modelName,
        const char* basePath,
        const int version,
        const char* loaderOptions,
        char** xmlBuffer, int* xmlLen,
        char** binBuffer, int* binLen) = 0;

    /**
         * @brief Get the model black list status
         *
         * @param model name for which black list status is required
         * @param version for which the black list status is required
         * @return blacklist status
         */
    virtual bool getModelBlacklistStatus(const char* modelName, int version) {
        return false;
    };

    /**
         * @brief Unload model resources by custom loader once model is unloaded by OVMS
         *
         * @param model name which is been unloaded
         * @param version which is been unloaded
         * @return status
         */
    virtual int unloadModel(const char* modelName, int version) = 0;

    /**
         * @brief Deinitialize the custom loader
         *
         */
    virtual void loaderDeInit() = 0;
};

// the types of the class factories
typedef CustomLoaderInterface* create_t();
typedef void destroy_t(CustomLoaderInterface*);

}  // namespace ovms
#endif
