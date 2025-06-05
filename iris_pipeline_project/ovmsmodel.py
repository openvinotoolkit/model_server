import pandas as pd
import numpy as np
import joblib
import os
from pyovms import Tensor

class OvmsPythonModel:
    def initialize(self, kwargs):
        print("Initializing model")
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        self.model = joblib.load(model_path)
        print("Model loaded successfully.")

    def execute(self, inputs):
        print("Executing inference...")
        input_tensor = inputs[0]
        input_data = input_tensor.as_numpy()
        
        csv_data = input_data.tobytes().decode('utf-8')
        df = pd.read_csv(pd.compat.StringIO(csv_data))
        
        features = df.iloc[:, :-1]  
        preds = self.model.predict(features)
        
        output = Tensor.from_numpy(np.array(preds, dtype=np.int32))
        return [output]

    def finalize(self):
        print("Finalizing model")
