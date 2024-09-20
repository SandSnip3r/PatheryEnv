CXX = g++
CXXFLAGS = -Wall -fPIC -std=c++17
LDFLAGS = -shared

TARGET = pathfinding.so

all: $(TARGET)

$(TARGET): pathfinder.o pathfinding.o
	$(CXX) $(LDFLAGS) -o $@ $^

pathfinding.o: pathfinding.cpp pathfinder.hpp
	$(CXX) $(CXXFLAGS) -c pathfinding.cpp

pathfinder.o: pathfinder.cpp pathfinder.hpp
	$(CXX) $(CXXFLAGS) -c pathfinder.cpp

clean:
	rm -f *.o $(TARGET)