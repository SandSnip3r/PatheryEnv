CXX = g++
CXXFLAGS = -Wall -fPIC -std=c++17
LDFLAGS = -shared

TARGET = pathfinding.so

all: $(TARGET)

$(TARGET): pathfinding.o
	$(CXX) $(LDFLAGS) -o $@ $^

pathfinding.o: pathfinding.cpp
	$(CXX) $(CXXFLAGS) -c pathfinding.cpp

clean:
	rm -f *.o $(TARGET)