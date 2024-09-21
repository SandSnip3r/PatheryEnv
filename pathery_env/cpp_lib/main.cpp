#include "pathfinding.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>

#define UNROLL_X2(statement) \
do {\
  statement; \
  statement; \
} while(false)

int main() {
  constexpr const int32_t kGridHeight = 19;
  constexpr const int32_t kGridWidth = 27;
  constexpr const int32_t grid[kGridHeight*kGridWidth] = {
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 17, 0, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 22, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 16, 18, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 5, 0, 19, 0, 0, 15, 0, 0, 0, 0, 8, 0, 0, 0, 9, 1, 21, 5, 0, 4,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 5, 0, 0, 11, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 20, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
  };
  constexpr const int32_t kCheckpointCount = 9;
  constexpr const int32_t kTeleporterCount = 4;
  constexpr const int32_t kOutputBufferSize = 10261;
  int32_t outputBuffer[kOutputBufferSize];

  // Infinite while for profiling
  while (1) {
    getShortestPath(grid, kGridHeight, kGridWidth, kCheckpointCount, kTeleporterCount, outputBuffer, kOutputBufferSize);
  }

  // // Timed with print
  // const auto startTime = std::chrono::high_resolution_clock::now();
  // UNROLL_X2(UNROLL_X2(UNROLL_X2(UNROLL_X2(UNROLL_X2(UNROLL_X2(UNROLL_X2(getShortestPath(grid, kGridHeight, kGridWidth, kCheckpointCount, kTeleporterCount, outputBuffer, kOutputBufferSize))))))));
  // const auto endTime = std::chrono::high_resolution_clock::now();
  // std::cout << "Took " << std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count()/1000.0 << "ms" << std::endl;
  return 0;
}