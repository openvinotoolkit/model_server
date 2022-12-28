docker build . -t go-client

mkdir build
for i in ./*.go
do 
  docker run -v $(pwd)/build:/app/build go-client go build -o /app/build $i 
done