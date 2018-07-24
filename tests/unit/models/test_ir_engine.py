from ie_serving.models.ir_engine import IrEngine


def test_init_class():
    model_xml = 'model1.xml'
    model_bin = 'model1.bin'
    exec_net = None
    input_key = 'input'
    inputs = {input_key: []}
    outputs = None
    engine = IrEngine(model_bin=model_bin, model_xml=model_xml,
                      exec_net=exec_net, inputs=inputs, outputs=outputs)
    assert model_xml == engine.model_xml
    assert model_bin == engine.model_bin
    assert exec_net == engine.exec_net
    assert input_key == engine.input_blob
    assert inputs == engine.inputs
    assert outputs == engine.outputs
