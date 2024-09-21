#ifndef PATHFINDER_HPP_
#define PATHFINDER_HPP_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

enum class CellType {
  kOpen = 0,
  kRock = 1,
  kWall = 2,
  kStart = 3,
  kGoal = 4,
  kIce = 5,
  kLength
};

struct Position {
  Position() = default;
  Position(int r, int c) : row(r), col(c) {}
  int32_t row, col;
};

// Define std::hash for Position.
namespace std {
template<>
struct hash<Position> {
  std::size_t operator()(const Position &position) const {
    return std::size_t(position.row) << 32 | position.col;
  }
};
} // namespace std

bool operator<(const Position &p1, const Position &p2);
bool operator==(const Position &p1, const Position &p2);

class Pathfinder {
public:
  Pathfinder(const int32_t *grid, int32_t height, int32_t width, int32_t checkpointCount, int32_t teleporterCount);
  std::vector<Position> calculateShortestPath() const;

private:
  using TeleporterIndexType = int;
  const int32_t *const grid_;
  const int32_t gridHeight_, gridWidth_;
  const int32_t checkpointCount_;
  const int32_t teleporterCount_;
  std::vector<Position> startPositions_;
  std::map<TeleporterIndexType, std::pair<std::set<Position>, std::set<Position>>> teleporterInfo_;

  void adjustPathForTeleporters(const int destinationType, std::set<int> &usedTeleporters, std::vector<Position> &path) const;
  std::vector<Position> calculateShortestSubpath(const Position &startPosition, const int destinationType) const;

  template<typename StartPositionsContainerType>
  std::vector<Position> calculateShortestPathFromMultipleStarts(const StartPositionsContainerType &startPositions, const int destinationType) const {
    std::vector<Position> bestPath;
    // Calculate shortest path starting from each start position and choose the shortest one that is not empty.
    for (const Position &startPosition : startPositions) {
      std::vector<Position> path = calculateShortestSubpath(startPosition, destinationType);
      if (!path.empty()) {
        // We found a path.
        if (bestPath.empty()) {
          // No best exists yet, save this.
          bestPath = std::move(path);
        } else {
          if (path.size() < bestPath.size()) {
            // Found a new best path
            bestPath = std::move(path);
          }
        }
      }
    }
    return bestPath;
  }

  int posToLinear(int row, int col) const {
    return row * gridWidth_ + col;
  }
};

#endif // PATHFINDER_HPP_