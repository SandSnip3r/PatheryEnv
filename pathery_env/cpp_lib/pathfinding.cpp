#include <iostream>

int posToLinear(int row, int col, int width) {
  return row * width + col;
}

extern "C" {

void getShortestPath(int32_t *grid, int32_t height, int32_t width) {
  std::cout << "Getting shortest path on grid of size " << height << 'x' << width << std::endl;
  for (int row=0; row<height; ++row) {
    for (int col=0; col<width; ++col) {
      std::cout << grid[posToLinear(row, col, width)] << " ";
    }
    std::cout << std::endl;
  }
}

}