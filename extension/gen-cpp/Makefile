CXX=g++
CFLAGS=-fPIC -O3
CXXFLAGS=$(CFLAGS)
AR=ar

all: libHbase.a 

libHbase.a: Hbase.o Hbase_types.o Hbase_constants.o
	mkdir -p deploy/lib deploy/include
	cp Hbase_constants.h Hbase.h Hbase_types.h deploy/include
	$(AR) rcs deploy/lib/$@ $^

..cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -fr demo deploy *.o *~

demo: DemoClient.cpp Makefile
	g++ $< -O3 -Ideploy/include -I/usr/local/include/thrift -L/usr/local/lib -Ldeploy/lib -lHbase -lthrift -o $@

