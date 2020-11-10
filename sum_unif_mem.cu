#include <iostream>

using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

__global__ void sum(float *a, float *b, float *c, int N)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int threadsNum = blockDim.x*gridDim.x;
    for (int i = id; i < N; i+=threadsNum)
        c[i] = a[i]+b[i];
}

int main(void)
{
    int N = 10*1000*1000;
    float *a, *b, *c, *c_check;

    CHECK( cudaMallocManaged(&a, N*4) );
    CHECK( cudaMallocManaged(&b, N*4) );
    CHECK( cudaMallocManaged(&c, N*4) );
    c_check = new float[N];
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = 2*i;
    }
    for (int i = 0; i < N; i++) c_check[i] = a[i] + b[i];

    sum<<<1024, 1024>>>(a, b, c, N);

    CHECK( cudaDeviceSynchronize(); );

    // check
    for (int i = 0; i < N; i++)
        if (abs(c[i] - c_check[i]) > 1e-6)
        {
            cout << "Error in element N " << i << ": c[i] = " << c[i]
                 << " c_check[i] = " << c_check[i] << "\n";
            exit(1);
        }
    CHECK( cudaFree(a) );
    CHECK( cudaFree(b) );
    CHECK( cudaFree(c) );
    return 0;
}
