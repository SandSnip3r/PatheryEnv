#include "pathfinder.hpp"
#include "pathfinding.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

void getShortestPath(const int32_t *grid, int32_t height, int32_t width, int32_t checkpointCount, int32_t teleporterCount, int32_t *output, int32_t outputBufferSize) {
  // Construct Pathfinder
  Pathfinder pathfinder(grid, height, width, checkpointCount, teleporterCount);

  // Get shortest path
  const std::vector<Position> shortestPath = pathfinder.calculateShortestPath();

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