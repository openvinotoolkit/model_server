# Custom node for OCR implementation with east-resnet50 and crnn models 

This custom node analysis the response of east-resnet50 model. Based on the inference results and the original image,
it generates a list of detected boxes for text recognition.
Each image in the output will be resized to the predefined target size to make it fit the next inference model in the 
DAG pipeline.

# Building custom node library

You can build the shared library of the custom node simply by command in this custom node folder context:
```
make
```
It will compile the library inside a docker container and save the results in `lib` folder.

# Custom node inputs

# Custom node outputs

# Custom node parameters
