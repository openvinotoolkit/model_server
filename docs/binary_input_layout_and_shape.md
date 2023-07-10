# Input Shape and Layout Considerations{#ovms_docs_binary_input_layout_and_shape}

Before processing in the target AI model, binary image data is encoded by OVMS to a NHWC layout in BGR color format.
It is also resized to the model or pipeline node resolution. When the model resolution supports range of values and image data shape is out of range it will be adjusted to the nearer border. For example, when model shape is: [1,100:200,200,3]:

- if input shape is [1,90,200,3] it will be resized into [1,100,200,3]
- if input shape is [1,220,200,3] it will be resized into [1,200,200,3]

In order to use binary input functionality, model or pipeline input layout needs to be compatible with `N...HWC` and have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions. It means that input layout needs to resemble `NHWC` layout, e.g. default `N...` will work. On the other hand, binary image input is not supported for inputs with `NCHW` layout. 

To fully utilize binary input utility, automatic image size alignment will be done by OVMS when:
- input shape does not include dynamic dimension value (`-1`)
- input layout is configured to be either `...` (custom nodes) and `NHWC` or `N?HWC` (or `N?HWC`, when modified by a [demultiplexer](demultiplexing.md))

Processing the binary image requests requires the model or the custom nodes to accept BGR color 
format with data with the data range from 0-255. Original layout of the input data can be changed in the 
OVMS configuration in runtime. For example when the original model has input shape [1,3,224,224] add a parameter
in the OVMS configuration "layout": "NHWC:NCHW" or the command line parameter `--layout NHWC:NCHW`. In result, the model will
have effective shape [1,224,224,3] and layout `NHWC`.

In case the model was trained with RGB color format and a range other than 0-255, the [Model Optimizer](tf_model_binary_input.md) can apply the required adjustments:
  
`--reverse_input_channels`: Switch the input channels order from RGB to BGR (or vice versa). Applied to original inputs of the model **only** if a number of channels equals 3. Applied after application of --mean_values and --scale_values options, so numbers in --mean_values and  --scale_values go in the order of channels used in the original model  
`--scale` : All input values coming from original network inputs  will be divided by this value. When a list of inputs  is overridden by the --input parameter, this scale is  not applied for any input that does not match with the  original input of the model  
`--mean_values` :  Mean values to be used for the input image per  channel. Values to be provided in the (R,G,B) or (B,G,R) format. Can be defined for desired input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained.

Blob data precision from binary input decoding is set automatically based on the target model or the [DAG pipeline](dag_scheduler.md) node.