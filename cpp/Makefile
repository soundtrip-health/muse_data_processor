CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
TARGET    = museproc
OBJS      = main.o jsonl_reader.o metrics.o

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

main.o: main.cpp jsonl_reader.h metrics.h ring_buffer.h
jsonl_reader.o: jsonl_reader.cpp jsonl_reader.h ring_buffer.h
metrics.o: metrics.cpp metrics.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
