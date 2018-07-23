from ie_serving.config import CPU_EXTENSION, DEVICE, PLUGIN_DIR
from openvino.inference_engine import IENetwork, IEPlugin


class IrEngine():

    def __init__(self, model_xml, model_bin, exec_net, inputs, outputs):
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.exec_net = exec_net
        self.input_blob = next(iter(inputs))
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def build(cls, model_xml, model_bin, num_request: int=1):
        plugin = IEPlugin(device=DEVICE, plugin_dirs=PLUGIN_DIR)
        if CPU_EXTENSION and 'CPU' in DEVICE:
            plugin.add_cpu_extension(CPU_EXTENSION)
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)
        exec_net = plugin.load(network=net, num_requests=num_request)
        inputs = net.inputs
        outputs = net.outputs
        ir_engine = cls(model_xml=model_xml, model_bin=model_bin,
                        exec_net=exec_net, inputs=inputs, outputs=outputs)
        return ir_engine

    def infer(self, data: dict):
        results = self.exec_net.infer(inputs=data)
        return results
