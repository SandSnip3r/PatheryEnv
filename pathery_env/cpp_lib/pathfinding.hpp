#ifndef PATHFINDING_HPP_
#define PATHFINDING_HPP_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void getShortestPath(const int32_t *grid, int32_t height, int32_t width, int32_t checkpointCount, int32_t teleporterCount, int32_t *output, int32_t outputBufferSize);

#ifdef __cplusplus
}
#endif

#endif // PATHFINDING_HPP_