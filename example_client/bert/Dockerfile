FROM centos:7
RUN yum install -y python3-pip
WORKDIR /bert
COPY bert_question_answering_demo_ovms.py .
COPY requirements.txt .
ADD https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/demos/python_demos/common/tokens_bert.py .
ADD https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/demos/python_demos/common/html_reader.py .
ADD https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/bert-large-uncased-whole-word-masking-squad-int8-0001/vocab.txt .
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "bert_question_answering_demo_ovms.py", "-v", "vocab.txt", "-i", "https://en.wikipedia.org/wiki/Graph_theory", "--question", "what is a graph", "--grpc_port", "4000", "--input_names", "result.1,result.2,result.3", "--output_names", "5211,5212"]