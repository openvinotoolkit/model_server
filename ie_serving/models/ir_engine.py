from ie_serving.config import CPU_EXTENSION, DEVICE, PLUGIN_DIR
from inference_engine import IENetwork, IEPlugin


class IrEngine():

    def __init__(self, model_xml, model_bin):
        self.model_xml = model_xml
        self.model_bin = model_bin
        plugin = IEPlugin(device=DEVICE, plugin_dirs=PLUGIN_DIR)
        if CPU_EXTENSION and 'CPU' in DEVICE:
            plugin.add_cpu_extension(CPU_EXTENSION)
        net = IENetwork.from_ir(model=self.model_xml, weights=self.model_bin)
        self.exec_net = plugin.load(network=net, num_requests=1)
        self.input_blob = next(iter(net.inputs))
        self.inputs = net.inputs
        self.outputs = net.outputs

    def infer(self, data):
        results = self.exec_net.infer(inputs={self.input_blob: data})
        return results
