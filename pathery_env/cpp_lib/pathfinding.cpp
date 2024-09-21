#include "pathfinder.hpp"
#include "pathfinding.hpp"

#include <cstdint>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

void getShortestPath(int32_t *grid, int32_t height, int32_t width, int32_t checkpointCount, int32_t teleporterCount, int32_t *output, int32_t outputBufferSize) {
  // auto startTime = std::chrono::high_resolution_clock::now();
  // Construct Pathfinder
  Pathfinder pathfinder(grid, height, width, checkpointCount, teleporterCount);
  // auto midTime = std::chrono::high_resolution_clock::now();

  // Get shortest path
  const std::vector<Position> shortestPath = pathfinder.calculateShortestPath();
  // auto afterTime = std::chrono::high_resolution_clock::now();
  // const double pathfindingDuration = std::chrono::duration_cast<std::chrono::microseconds>(afterTime-midTime).count()/1000.0;
  // const double totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(afterTime-startTime).count()/1000.0;
  // std::cout << "Spent " << pathfindingDuration << "ms pathfinding (" << totalDuration << " ms) total" << std::endl;

  // Serialize the shortest path into the output buffer
  output[0] = shortestPath.size();
  int i=1;
  for (const Position &position : shortestPath) {
    if (i+1 >= outputBufferSize) {
      throw std::runtime_error("Overflow! Output buffer is not large enough for path. Path length is "+std::to_string(shortestPath.size())+", buffer can only hold "+std::to_string((outputBufferSize-1)/2)+" items");
    }
    output[i] = position.row;
    output[i+1] = position.col;
    i += 2;
  }
}

}