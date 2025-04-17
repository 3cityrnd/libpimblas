#pragma once

#define RET_TEST_FAIL exit(EXIT_FAILURE)
#define RET_TEST_OK exit(EXIT_SUCCESS)

template <typename T>
struct AlignedAllocator {
  typedef T value_type;

  AlignedAllocator() = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U> &) {}

  T *allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }

    void *ptr = nullptr;
    if (posix_memalign(&ptr, 8, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  void deallocate(T *ptr, std::size_t n) { free(ptr); }
};

namespace pimblas {

template <class T>
using vector = std::vector<T, AlignedAllocator<T>>;

}

template <typename T>
pimblas::vector<T> generateRandomIntegral(size_t size, T min, T max) {
  // std::static_assert(std::is_integral_v<T>, "T must be integral");
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  pimblas::vector<T> randomNumbers(size);
  std::for_each(randomNumbers.begin(), randomNumbers.end(), [&dis, &gen](T &v) { v = dis(gen); });
  return randomNumbers;
}

pimblas::vector<int> generateRandomIntegers(int size, int min, int max) {
  show_debug("Generate Random Ints range {} - {}  size={}", min, max, size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  pimblas::vector<int> randomNumbers(size);
  std::for_each(randomNumbers.begin(), randomNumbers.end(), [&dis, &gen](int &v) { v = dis(gen); });
  return randomNumbers;
}

pimblas::vector<float> generateRandomFloats(size_t size, float min, float max) {
  show_debug("Generate Random Floats range {} - {}  size={}", min, max, size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  pimblas::vector<float> randomNumbers(size);
  std::for_each(randomNumbers.begin(), randomNumbers.end(), [&dis, &gen](float &v) { v = dis(gen); });
  return randomNumbers;
}

pimblas::vector<float> generateAscendingFloats(size_t size, float start, float step) {
  pimblas::vector<float> numbers(size);
  float val = start;
  for (auto &ele : numbers) {
    ele = val;
    val += step;
  }
  return numbers;
}

pimblas::vector<int> generate(int size, int fill = 0) {
  pimblas::vector<int> v(size);
  std::fill(v.begin(), v.end(), fill);
  return v;
}

template <class V>
bool same_vectors(const V &v1, const V &v2) {
  return std::equal(v1.begin(), v1.end(), v2.begin());
}
template <typename T>
bool same(T *bufferA, T *bufferB, size_t size) {
  bool valid = true;
  for (size_t i = 0; i < size; i++) {
    if (bufferA[i] != bufferB[i]) {
      std::cout << bufferA[i] << " " << bufferB[i] << " differ at " << i << ".\n";
      valid = false;
    }
  }
  return valid;
}

template <typename T>
bool mostly_same_rel(T *bufferA, T *bufferB, size_t size, T relTolerance) {
  bool valid = true;
  for (size_t i = 0; i < size; i++) {
    auto diff = std::abs(bufferA[i] - bufferB[i]);
    auto tolerance = relTolerance * std::max(std::abs(bufferA[i]), std::abs(bufferB[i]));
    if (diff > tolerance) {
      std::cout << bufferA[i] << " " << bufferB[i] << " differ at " << i << ".\n";
      valid = false;
    }
  }
  return valid;
}

template <typename T>
bool mostly_same_abs(T *bufferA, T *bufferB, size_t size, T absTolerance) {
  bool valid = true;
  for (size_t i = 0; i < size; i++) {
    auto diff = std::abs(bufferA[i] - bufferB[i]);
    if (diff > absTolerance) {
      std::cout << bufferA[i] << " " << bufferB[i] << " differ at " << i << ".\n";
      valid = false;
    }
  }
  return valid;
}
