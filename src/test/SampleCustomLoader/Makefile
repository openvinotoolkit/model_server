OUT_LIB = libsampleloader.so

LIB_INC = -Iinclude -I../../ -Irapidjson-1.1.0/include
CFLAGS = -fPIC -Wl,-rpath,. -fpic 
LDFLAGS = -ldl -lpthread

all:
	wget https://github.com/Tencent/rapidjson/archive/v1.1.0.zip
	unzip v1.1.0.zip

	g++ -c -g -fPIC $(LIB_INC) *.cpp 
	g++ *.o -g $(CFLAGS) $(LDFLAGS) $(LIB_INC) -shared -o $(OUT_LIB) 
	$(RM) *.o
	$(RM) v1.1.0.zip
	$(RM) rapidjson-1.1.0 -r

clean:
	$(RM) $(OUT_LIB) *.o
	$(RM) v1.1.0.zip
	$(RM) rapidjson-1.1.0 -r
