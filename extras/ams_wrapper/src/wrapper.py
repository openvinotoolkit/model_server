from cheroot.wsgi import Server as WSGIServer, PathInfoDispatcher
from api.dispatcher import create_dispatcher
from config import AVAILABLE_MODELS

def start_rest_service(port, num_threads=1):
    dispatcher = PathInfoDispatcher({'/': create_dispatcher(AVAILABLE_MODELS)})
    server = WSGIServer(('0.0.0.0', port), dispatcher,
                        numthreads=num_threads)
    print(f"Server will start listetning on port {port}")
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()

start_rest_service(5000)
