
def make_tensor_proto(values, dtype=None, shape=None, use_tensor_content=True):
    '''
    Create TensorProto object from values.

    Args:

        values: Values to put in the TensorProto.
            The accepted types are: python scalar, pythons list, numpy scalar and numpy ndarray.
            Python scalars and lists are internally converted to their numpy counterparts before further processing.
            String values are converted to bytes and placed in TensorProto string_val field. 

        dtype (optional): tensor_pb2 DataType value. 
            If not provided, the function will try to infer the data type from **values** argument.

        shape (optional): The list of integers defining the shape of the tensor. 
            If not provided, the function will try to infer the shape from **values** argument.

        use_tensor_content (optional): Determines if the **values** should be placed in tensor_content field of TensorProto.  
            This is applicable for numeric types. Strings are always placed in TensorProto string_val field.
            The default value of this argument is True.

    Returns:
        TensorProto object filled with **values**.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:
        With python string scalar:

        >>> data = "hello"
        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)

        With python list:

        >>> data = [[1, 2, 3], [4, 5, 6]]
        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)

        With numpy array:

        >>> data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)
    '''

    raise NotImplementedError

def make_ndarray(tensor_proto):
    '''
    Create numpy ndarray from tensor_proto.

    Args:
        tensor_proto: TensorProto object.

    Returns:
        Numpy ndarray with tensor contents.

    Raises:
        TypeError:  if unsupported type is provided.

    Examples:
        Create TensorProto with make_tensor_proto and convert it back to numpy array with make_ndarray:

        >>> data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        >>> print(data)

        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)

        >>> output = make_ndarray(tensor_proto)
        >>> print(output)
    '''
    
    raise NotImplementedError