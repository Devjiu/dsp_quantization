#include <complex>
#include <iostream>
#include <valarray>
#include <vector>

#include "third-party/xtensor/include/xtensor/xio.hpp"
#include "third-party/xtensor/include/xtensor/xrandom.hpp"
#include "third-party/xtensor/include/xtensor/xtensor.hpp"
#include "third-party/matplotlib-cpp/matplotlibcpp.h"

const double PI = 3.141592653589793238460;

using Complex = std::complex<double>;
using CArray = std::valarray<Complex>;
namespace plt = matplotlibcpp;

namespace vanilla {
struct Dims {
  bool operator==(Dims &&other) const {
    if (dims_.size() != other.getNumDims())
      return false;
    for (uint32_t dim_ind = 0; dim_ind < dims_.size(); ++dim_ind) {
      if (dims_[dim_ind] != other.getDim(dim_ind))
        return false;
    }
    return true;
  }

  uint32_t getDim(uint32_t index) { return dims_.at(index); }

  uint32_t getNumDims() const { return dims_.size(); }

private:
  std::vector<uint32_t> dims_;
};

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive
void fft(CArray &x) {
  const size_t N = x.size();
  if (N <= 1)
    return;

  // divide
  CArray even = x[std::slice(0, N / 2, 2)];
  CArray odd = x[std::slice(1, N / 2, 2)];

  // conquer
  fft(even);
  fft(odd);

  // combine
  for (size_t k = 0; k < N / 2; ++k) {
    Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
    x[k] = even[k] + t;
    x[k + N / 2] = even[k] - t;
  }
}

/*
void fft(xt::xtensor<float, > &x) {
  const size_t N = x.size();
  if (N <= 1)
    return;

  // divide
  CArray even = x[std::slice(0, N / 2, 2)];
  CArray odd = x[std::slice(1, N / 2, 2)];

  // conquer
  fft(even);
  fft(odd);

  // combine
  for (size_t k = 0; k < N / 2; ++k) {
    Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
    x[k] = even[k] + t;
    x[k + N / 2] = even[k] - t;
  }
}
*/

} // namespace vanilla

namespace opt {
// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive
// !!! Warning : in some cases this code make result different from not
// optimased version above (need to fix bug) The bug is now fixed @2017/05/30
void fft(CArray &x) {
  // DFT
  unsigned int N = x.size(), k = N, n;
  double thetaT = 3.14159265358979323846264338328L / N;
  Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
  while (k > 1) {
    n = k;
    k >>= 1;
    phiT = phiT * phiT;
    T = 1.0L;
    for (unsigned int l = 0; l < k; l++) {
      for (unsigned int a = l; a < N; a += n) {
        unsigned int b = a + k;
        Complex t = x[a] - x[b];
        x[a] += x[b];
        x[b] = t * T;
      }
      T *= phiT;
    }
  }
  // Decimate
  unsigned int m = (unsigned int)log2(N);
  for (unsigned int a = 0; a < N; a++) {
    unsigned int b = a;
    // Reverse bits
    b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
    b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
    b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
    b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
    b = ((b >> 16) | (b << 16)) >> (32 - m);
    if (b > a) {
      Complex t = x[a];
      x[a] = x[b];
      x[b] = t;
    }
  }
  //// Normalize (This section make it not working correctly)
  // Complex f = 1.0 / sqrt(N);
  // for (unsigned int i = 0; i < N; i++)
  //	x[i] *= f;
}
} // namespace opt

// inverse fft (in-place)
void ifft(CArray &x) {
  // conjugate the complex numbers
  x = x.apply(std::conj);

  // forward fft
  vanilla::fft(x);

  // conjugate the complex numbers again
  x = x.apply(std::conj);

  // scale the numbers
  x /= x.size();
}

int main() {
  const Complex test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
  CArray data(test, 8);

  /*arma::Cube<float> arm_mat;
  arm_mat.randu(5, 10, 3);
  arm_mat.print("arm_works: ");*/

  // forward fft
  vanilla::fft(data);

  std::cout << "fft" << std::endl;
  for (int i = 0; i < 8; ++i) {
    std::cout << data[i] << std::endl;
  }

  std::vector<float> time(data.size());
  std::generate(time.begin(), time.end(), [](){
    static int i = 1;
    return i++;
  });
  std::vector<float> show_data;
  std::vector<Complex> data_v;
  data_v.assign(std::begin(data), std::end(data));
  std::transform(data_v.begin(), data_v.end(), std::back_inserter(show_data),
                 [](Complex const& x) { return x.real(); });

  std::cout << "x sz: " << show_data.size() << " t: " << time.size() << std::endl;
  plt::plot(show_data, time);
  plt::save("fft_after.png");

  // inverse fft
  ifft(data);
  show_data.clear();
  data_v.assign(std::begin(data), std::end(data));
  std::transform(data_v.begin(), data_v.end(), std::back_inserter(show_data),
                 [](Complex const& x) { return x.real(); });
  plt::plot(show_data, time);
  plt::save("ifft_after.png");

  std::cout << std::endl << "ifft" << std::endl;
  for (int i = 0; i < 8; ++i) {
    std::cout << data[i] << std::endl;
  }

  xt::xtensor<double, 2>::shape_type shape = {2, 3};
  xt::xtensor<double, 2> a0(shape);
  xt::xtensor<double, 2> a1(shape, 2.5);
  xt::xtensor<double, 2> a2 = {{1., 2., 3.}, {4., 5., 6.}};
  auto a3 = xt::xtensor<double, 2>::from_shape(shape);

  xt::random::seed(0);
  xt::xtensor<uint32_t, 4> d4 =
      xt::random::randint<uint32_t>({1, 3, 4, 5}, 0, 20);

  std::cout << "xtensor out: " << std::endl;
  std::cout << "\t" << d4 << std::endl;
  return 0;
}