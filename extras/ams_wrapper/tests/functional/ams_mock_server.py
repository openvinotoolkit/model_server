from cheroot.wsgi import Server as WSGIServer, PathInfoDispatcher
import falcon


class MockEntityObjectDetectionModel(object):
    def on_post(self, req, resp):
        body = req.stream.read()
        print('---')
        print(type(body))
        print(req.headers)
        print('---')
        if (isinstance(body, bytes) and len(body) > 0 and 
            req.headers.get('CONTENT-TYPE') in {'image/png', 'image/jpg', 'image/bmp'}):

            resp.status = falcon.HTTP_200
            resp.body = '''
            {"inferences": [
                {        
                    "type": "entity",
                    "subtype": "objectDetection",    
                    "entity":
                    {
                      "tag": { "value": "dog", "confidence": 0.97 },
                      "box": { "l": 0.0, "t": 0.0, "w": 0.0, "h": 0.0 }
                    }
                }
            ]}
            '''
        else:
            resp.status = falcon.HTTP_400

app = falcon.API()

model = MockEntityObjectDetectionModel()
app.add_route('/entity-object-detection', model)

d = PathInfoDispatcher({'/': app})
server = WSGIServer(('0.0.0.0', 8000), d)
server.start()