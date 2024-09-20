#include "pathfinder.hpp"

#include <algorithm>
#include <optional>
#include <queue>

Pathfinder::Pathfinder(int32_t *grid, int32_t height, int32_t width, int32_t checkpointCount, int32_t teleporterCount) : grid_(grid), gridHeight_(height), gridWidth_(width), checkpointCount_(checkpointCount), teleporterCount_(teleporterCount) {
  // Extract start positions from the grid.
  // Also extract teleporter info from the grid.
  const int teleporterStartValue = static_cast<int>(CellType::kLength) + checkpointCount_;
  for (int row=0; row<height; ++row) {
    for (int col=0; col<width; ++col) {
      const int cellValue = grid[posToLinear(row, col)];
      if (cellValue == static_cast<int>(CellType::kStart)) {
        startPositions_.emplace_back(row,col);
      } else if (cellValue >= teleporterStartValue) {
        // Is a teleporter cell.
        const TeleporterIndexType teleporterIndex = (cellValue - teleporterStartValue) / 2;
        const Position position(row, col);
        auto &teleporterInfo = teleporterInfo_[teleporterIndex];
        if ((cellValue - teleporterStartValue) % 2 == 0) {
          // Is an IN
          teleporterInfo.first.insert(position);
        } else {
          // Is an OUT
          teleporterInfo.second.insert(position);
        }
      }
    }
  }
}

std::vector<Position> Pathfinder::calculateShortestPath() const {
  std::set<TeleporterIndexType> usedTeleporters;

  int firstDestinationType;
  if (checkpointCount_ == 0) {
    // No checkpoints, path directly from the start to the goal.
    firstDestinationType = static_cast<int>(CellType::kGoal);
  } else {
    // First, path to the first checkpoint.
    firstDestinationType = static_cast<int>(CellType::kLength);
  }

  std::vector<Position> overallPath = calculateShortestPathFromMultipleStarts(startPositions_, firstDestinationType);
  adjustPathForTeleporters(firstDestinationType, usedTeleporters, overallPath);

  if (checkpointCount_ == 0) {
    // No checkpoints; done
    return overallPath;
  }

  if (overallPath.empty()) {
    // Since the first part of the path is blocked, the whole path is blocked.
    return overallPath;
  }

  auto extendPath = [&](int destinationType) -> bool {
    std::vector<Position> subPath = calculateShortestSubpath(overallPath.back(), destinationType);
    adjustPathForTeleporters(destinationType, usedTeleporters, subPath);
    if (subPath.empty()) {
      // If any sub-path is blocked, the entire path is blocked.
      return false;
    }
    overallPath.insert(overallPath.end(), subPath.begin(), subPath.end());
    return true;
  };

  // There are checkpoints. Our current path is up to the first one. Path to the next ones, if there are any.
  // Note: Pathery does not support missing checkpoints. For example, if the only checkpoints are A and C, pathing will fail.
  for (int i=1; i<checkpointCount_; ++i) {
    const int checkpointValue = static_cast<int>(CellType::kLength)+i;
    bool success = extendPath(checkpointValue);
    if (!success) {
      return {};
    }
  }

  // After the last checkpoint, finally path to the goal.
  bool success = extendPath(static_cast<int>(CellType::kGoal));
  if (!success) {
    return {};
  }

  return overallPath;
}

void Pathfinder::adjustPathForTeleporters(const int destinationType, std::set<int> &usedTeleporters, std::vector<Position> &path) const {
  // Takes a path and checks if it goes into any of the active teleporters. If it does, the path will be updated to go through the teleporter and find the new shortest path to the same destination type (maybe a different instance of the destination perviously found).
  // Does this path hit a teleporter?
  for (const auto &indexInfoPair : teleporterInfo_) {
    const TeleporterIndexType teleporterIndex = indexInfoPair.first;
    if (usedTeleporters.count(teleporterIndex) != 0) {
      // Already used this teleporter
      continue;
    }
    for (const Position &inPosition : indexInfoPair.second.first) {
      // Check if the path touches this unused teleporter IN.
      for (size_t pathIndex=0; pathIndex<path.size(); ++pathIndex) {
        const Position &pathPosition = path[pathIndex];
        if (pathPosition == inPosition) {
          // This path goes into a teleporter.
          usedTeleporters.insert(teleporterIndex);
          // Find the updated path from the best OUT of this teleporter to the closest destination.
          std::vector<Position> postTeleporterPath = calculateShortestPathFromMultipleStarts(indexInfoPair.second.second, destinationType);
          // Recurse, in case we go into another teleporter with the updated path.
          adjustPathForTeleporters(destinationType, usedTeleporters, postTeleporterPath);
          // Concatenate and return the path to the teleporter IN and the path after the teleporter OUT.
          path.erase(path.begin()+pathIndex+1, path.end());
          path.insert(path.end(), postTeleporterPath.begin(), postTeleporterPath.end());
          return;
        }
      }
    }
  }
}

std::vector<Position> Pathfinder::calculateShortestSubpath(const Position &startPosition, const int destinationType) const {
  std::set<Position> visited;
  std::set<Position> pushed;
  std::map<Position, Position> previous;
  enum class Direction {
    kUp,
    kRight,
    kDown,
    kLeft
  };
  struct PositionAndMaybeDirection {
    PositionAndMaybeDirection(const Position &pos) : position(pos) {}
    PositionAndMaybeDirection(const Position &pos, Direction dir) : position(pos), direction(dir) {}
    Position position;
    std::optional<Direction> direction;
  };
  std::queue<PositionAndMaybeDirection> bfsQueue;
  bfsQueue.emplace(startPosition);
  while (!bfsQueue.empty()) {
    const PositionAndMaybeDirection posAndDirection = bfsQueue.front();
    const Position &currentPosition = posAndDirection.position;
    const std::optional<Direction> &direction = posAndDirection.direction;

    // Check if we found the goal
    if (grid_[posToLinear(currentPosition.row, currentPosition.col)] == destinationType) {
      Position tmpCurrent = currentPosition;
      std::vector<Position> path = {tmpCurrent};
      int steps = 0;
      auto it = previous.find(tmpCurrent);
      while (it != previous.end()) {
        ++steps;
        tmpCurrent = it->second;
        path.push_back(tmpCurrent);
        it = previous.find(tmpCurrent);
      }
      // The starting position should not be in the path.
      path.pop_back();
      std::reverse(path.begin(), path.end());
      return path;
    }

    bfsQueue.pop();

    visited.emplace(currentPosition);

    auto pushNextPosition = [&](const Position &nextPosition, const Direction direction) {
      if (visited.count(nextPosition) != 0) {
        // Already visited
        return;
      }
      if (pushed.count(nextPosition) != 0) {
        // Already pushed
        return;
      }
      previous[nextPosition] = currentPosition;
      if (grid_[posToLinear(nextPosition.row, nextPosition.col)] == static_cast<int>(CellType::kIce)) {
        // On ice, push direction on queue
        bfsQueue.emplace(nextPosition, direction);
      } else {
        bfsQueue.emplace(nextPosition);
      }
      pushed.insert(nextPosition);
    };

    // Directions for moving: up, right, down, left (this is the order preferred by Pathery)
    if (currentPosition.row > 0 && (!direction || direction.value() == Direction::kUp)) {
      // Try up
      if (grid_[posToLinear(currentPosition.row-1, currentPosition.col)] != static_cast<int>(CellType::kRock) &&
          grid_[posToLinear(currentPosition.row-1, currentPosition.col)] != static_cast<int>(CellType::kWall)) {
        // Can move here
        pushNextPosition(Position(currentPosition.row-1, currentPosition.col), Direction::kUp);
      }
    }
    if (currentPosition.col < gridWidth_-1 && (!direction || direction.value() == Direction::kRight)) {
      // Try right
      if (grid_[posToLinear(currentPosition.row, currentPosition.col+1)] != static_cast<int>(CellType::kRock) &&
          grid_[posToLinear(currentPosition.row, currentPosition.col+1)] != static_cast<int>(CellType::kWall)) {
        // Can move here
        pushNextPosition(Position(currentPosition.row, currentPosition.col+1), Direction::kRight);
      }
    }
    if (currentPosition.row < gridHeight_-1 && (!direction || direction.value() == Direction::kDown)) {
      // Try down
      if (grid_[posToLinear(currentPosition.row+1, currentPosition.col)] != static_cast<int>(CellType::kRock) &&
          grid_[posToLinear(currentPosition.row+1, currentPosition.col)] != static_cast<int>(CellType::kWall)) {
        // Can move here
        pushNextPosition(Position(currentPosition.row+1, currentPosition.col), Direction::kDown);
      }
    }
    if (currentPosition.col > 0 && (!direction || direction.value() == Direction::kLeft)) {
      // Try left
      if (grid_[posToLinear(currentPosition.row, currentPosition.col-1)] != static_cast<int>(CellType::kRock) &&
          grid_[posToLinear(currentPosition.row, currentPosition.col-1)] != static_cast<int>(CellType::kWall)) {
        // Can move here
        pushNextPosition(Position(currentPosition.row, currentPosition.col-1), Direction::kLeft);
      }
    }
  }

  // No path found
  return {};
}

bool operator<(const Position &p1, const Position &p2) {
  if (p1.row == p2.row) {
    return p1.col < p2.col;
  }
  return p1.row < p2.row;
}

bool operator==(const Position &p1, const Position &p2) {
  return p1.row == p2.row && p1.col == p2.col;
}