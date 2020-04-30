#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


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
app.add_route('/vehicle-detection', model)

d = PathInfoDispatcher({'/': app})
server = WSGIServer(('0.0.0.0', 8000), d)
server.start()