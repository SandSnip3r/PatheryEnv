CXX = g++
CXXFLAGS = -Wall -fPIC -std=c++17 -O3
LDFLAGS = -shared

TARGET = pathfinding.so
EXECUTABLE = main

all: $(TARGET)

$(TARGET): pathfinder.o pathfinding.o
	$(CXX) $(LDFLAGS) -o $@ $^

pathfinding.o: pathfinding.cpp pathfinding.hpp pathfinder.hpp
	$(CXX) $(CXXFLAGS) -c pathfinding.cpp

pathfinder.o: pathfinder.cpp pathfinder.hpp
	$(CXX) $(CXXFLAGS) -c pathfinder.cpp

$(EXECUTABLE): main.o pathfinder.o pathfinding.o
	$(CXX) -o $@ $^

main.o: main.cpp pathfinder.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp

clean:
	rm -f *.o $(TARGET)