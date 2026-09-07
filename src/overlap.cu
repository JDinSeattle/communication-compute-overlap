// PortChannel setup follows MSCCL++'s MIT-licensed tutorial 04.
// Copyright (c) 2026 JDinSeattle. See THIRD_PARTY.md.
#include <cuda_runtime.h>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using Channel = mscclpp::PortChannelDeviceHandle;

__device__ uint32_t value(size_t i, uint32_t seq, int rank, int work) {
  uint32_t x = uint32_t(i) ^ (seq * 0x9e3779b9u) ^ (uint32_t(rank+1) * 0x85ebca6bu);
  for(int k=0;k<work;k++) x = x * 1664525u + 1013904223u;
  return x;
}

__global__ void exchange(Channel channel, uint32_t* memory, size_t count, size_t chunk,
                         uint32_t seq, int rank, int work, bool serial) {
  // A single CTA avoids non-cooperative grid barriers and occupancy-dependent deadlock.
  const size_t first = threadIdx.x;
  if(serial) {
    for(size_t i=first;i<count;i+=blockDim.x) memory[i]=value(i,seq,rank,work);
    // Every writing thread releases its writes before thread 0 publishes DMA work.
    __threadfence_system(); __syncthreads();
    if(threadIdx.x==0) channel.put(count*4, 0, count*4);
  } else {
    for(size_t off=0;off<count;off+=chunk) {
      size_t n = (chunk < count-off) ? chunk : count-off;
      for(size_t i=first;i<n;i+=blockDim.x) memory[off+i]=value(off+i,seq,rank,work);
      __threadfence_system(); __syncthreads();
      if(threadIdx.x==0) channel.put((count+off)*4, off*4, n*4);
      // Source slices remain immutable while the proxy/DMA overlaps subsequent compute.
      __syncthreads();
    }
  }
  if(threadIdx.x==0) {
    channel.signal();              // Remote publication ordered after all preceding puts.
    channel.flush(1000000000ll);    // Local completion: no DMA still owns the source.
    channel.wait(1000000000ll);     // Acquire peer publication before the host consumes data.
  }
  __syncthreads();
}

// The independent host reference composes an affine transform in O(log work),
// instead of copying the GPU loop; arithmetic is modulo 2^32 in both paths.
static uint32_t reference(size_t i, uint32_t seq, int rank, unsigned work) {
  uint32_t x = uint32_t(i) ^ (seq*0x9e3779b9u) ^ (uint32_t(rank+1)*0x85ebca6bu);
  uint32_t mul=1664525u, add=1013904223u, total_mul=1, total_add=0;
  while(work) {
    if(work&1) { total_add=mul*total_add+add; total_mul=mul*total_mul; }
    add=mul*add+add; mul=mul*mul; work>>=1;
  }
  return total_mul*x+total_add;
}

static long number(const char* input,long low,long high) {
  std::string s(input); size_t used=0; long v=std::stol(s,&used);
  if(used!=s.size() || v<low || v>high) throw std::invalid_argument("numeric argument out of range");
  return v;
}

__global__ void compute_only(uint32_t* output,size_t count,uint32_t seq,int rank,int work) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    output[i]=value(i,seq,rank,work);
}

static int self_test() {
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  uint32_t* output=nullptr;
  constexpr size_t max_count=16389;
  MSCCLPP_CUDATHROW(cudaMalloc(&output,max_count*4));
  unsigned cases=0;
  for(size_t count : {size_t(4),size_t(257),max_count})
    for(int work : {0,1,32,257})
      for(int rank : {0,1})
        for(uint32_t seq : {1u,2u,65537u}) {
          MSCCLPP_CUDATHROW(cudaMemset(output,0xa5,max_count*4));
          compute_only<<<4,256>>>(output,count,seq,rank,work);
          MSCCLPP_CUDATHROW(cudaGetLastError());
          std::vector<uint32_t> host(max_count);
          MSCCLPP_CUDATHROW(cudaMemcpy(host.data(),output,max_count*4,cudaMemcpyDeviceToHost));
          for(size_t i=0;i<count;i++) if(host[i]!=reference(i,seq,rank,unsigned(work)))
            throw std::runtime_error("GPU compute/reference mismatch");
          for(size_t i=count;i<max_count;i++) if(host[i]!=0xa5a5a5a5u)
            throw std::runtime_error("GPU compute wrote past logical extent");
          cases++;
        }
  MSCCLPP_CUDATHROW(cudaFree(output));
  std::cout<<"{\"scope\":\"single_gpu_compute_only\",\"status\":\"pass\",\"cases\":"<<cases
           <<",\"communication_tested\":false,\"reference\":\"host_affine_composition\"}\n";
  return 0;
}

static int preflight() {
  int n=0;auto err=cudaGetDeviceCount(&n);
  if(err!=cudaSuccess) {
    std::cout<<"{\"status\":\"blocked\",\"reason\":\"cuda_unavailable\",\"device_count\":0}\n";return 2;
  }
  int access01=0,access10=0;
  if(n>=2) { MSCCLPP_CUDATHROW(cudaDeviceCanAccessPeer(&access01,0,1)); MSCCLPP_CUDATHROW(cudaDeviceCanAccessPeer(&access10,1,0)); }
  bool okay=n>=2 && access01 && access10;
  std::cout<<"{\"status\":\""<<(okay?"ready":"blocked")<<"\",\"reason\":\""
           <<(okay?"two_gpu_p2p_available":"requires_two_peer_accessible_gpus")<<"\",\"device_count\":"<<n
           <<",\"p2p_0_to_1\":"<<access01<<",\"p2p_1_to_0\":"<<access10<<"}\n";
  return okay?0:2;
}

static int worker(int rank,int gpu,const std::string& endpoint,size_t bytes,size_t chunk_bytes,
                  int work,unsigned iterations,unsigned warmup,bool serial) {
  MSCCLPP_CUDATHROW(cudaSetDevice(gpu));
  auto bootstrap=std::make_shared<mscclpp::TcpBootstrap>(rank,2);
  bootstrap->initialize(endpoint);
  mscclpp::Communicator comm(bootstrap);
  const auto transport=mscclpp::Transport::CudaIpc;
  auto connection=comm.connect({transport,{mscclpp::DeviceType::GPU,gpu}},rank^1).get();
  auto semaphore=comm.buildSemaphore(connection,rank^1).get();
  mscclpp::GpuBuffer memory(2*bytes);
  auto local=comm.registerMemory(memory.data(),memory.bytes(),transport);
  comm.sendMemory(local,rank^1);
  auto remote=comm.recvMemory(rank^1).get();
  mscclpp::ProxyService proxy;
  auto sem_id=proxy.addSemaphore(semaphore);
  auto local_id=proxy.addMemory(local), remote_id=proxy.addMemory(remote);
  auto channel=proxy.portChannel(sem_id,remote_id,local_id);
  auto handle=channel.deviceHandle();
  cudaStream_t stream; cudaEvent_t start,stop;
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  MSCCLPP_CUDATHROW(cudaEventCreate(&start));MSCCLPP_CUDATHROW(cudaEventCreate(&stop));
  std::vector<uint32_t> received(bytes/4);
  proxy.startProxy();
  bool okay=true;
  try {
    for(unsigned seq=1;seq<=warmup+iterations;seq++) {
      // Includes the previous iteration's consumer acknowledgement. No buffer reuse before it.
      bootstrap->barrier();
      rusage before{},after{};getrusage(RUSAGE_SELF,&before);
      auto begin=std::chrono::steady_clock::now();
      MSCCLPP_CUDATHROW(cudaEventRecord(start,stream));
      exchange<<<1,256,0,stream>>>(handle,reinterpret_cast<uint32_t*>(memory.data()),bytes/4,chunk_bytes/4,seq,rank,work,serial);
      MSCCLPP_CUDATHROW(cudaGetLastError());
      MSCCLPP_CUDATHROW(cudaEventRecord(stop,stream));MSCCLPP_CUDATHROW(cudaEventSynchronize(stop));
      auto finish=std::chrono::steady_clock::now();getrusage(RUSAGE_SELF,&after);
      float gpu_ms=0;MSCCLPP_CUDATHROW(cudaEventElapsedTime(&gpu_ms,start,stop));
      MSCCLPP_CUDATHROW(cudaMemcpy(received.data(),reinterpret_cast<uint32_t*>(memory.data())+bytes/4,bytes,cudaMemcpyDeviceToHost));
      uint64_t checksum=1469598103934665603ull;
      for(size_t i=0;i<received.size();i++) {
        if(received[i]!=reference(i,seq,rank^1,unsigned(work))) {
          std::cerr<<"stale/corrupt data: rank="<<rank<<" seq="<<seq<<" index="<<i<<"\n";okay=false;break;
        }
        checksum=(checksum ^ received[i])*1099511628211ull;
      }
      if(!okay) break;  // Supervisor immediately terminates the peer, whose barrier would otherwise wait.
      if(seq>warmup) {
        auto cpu=[](const rusage& r){return r.ru_utime.tv_sec*1e6+r.ru_utime.tv_usec+r.ru_stime.tv_sec*1e6+r.ru_stime.tv_usec;};
        std::cout<<"{\"kind\":\"sample\",\"rank\":"<<rank<<",\"sequence\":"<<seq<<",\"bytes\":"<<bytes
                 <<",\"chunk_bytes\":"<<chunk_bytes<<",\"work\":"<<work<<",\"mode\":\""<<(serial?"serial":"overlap")
                 <<"\",\"gpu_us\":"<<gpu_ms*1000<<",\"wall_us\":"<<std::chrono::duration<double,std::micro>(finish-begin).count()
                 <<",\"process_cpu_us\":"<<cpu(after)-cpu(before)<<",\"checksum\":"<<checksum<<",\"correct\":true}\n";
      }
    }
    if(okay) bootstrap->barrier();
    proxy.stopProxy();
  } catch(...) { proxy.stopProxy(); throw; }
  MSCCLPP_CUDATHROW(cudaEventDestroy(start));MSCCLPP_CUDATHROW(cudaEventDestroy(stop));
  MSCCLPP_CUDATHROW(cudaStreamDestroy(stream));
  std::cout<<"{\"kind\":\"result\",\"rank\":"<<rank<<",\"status\":\""<<(okay?"pass":"fail")
           <<"\",\"transport\":\"CudaIpc\",\"channel\":\"PortChannel\"}\n";
  return okay?0:1;
}

int main(int argc,char** argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--preflight") return preflight();
    if(argc==2 && std::string(argv[1])=="--self-test") return self_test();
    if(argc!=10) { std::cerr<<"Usage: overlap rank gpu endpoint bytes chunk_bytes work iterations warmup serial|overlap\n";return 2; }
    int rank=number(argv[1],0,1),gpu=number(argv[2],0,1);
    if(gpu!=rank) throw std::invalid_argument("v1 requires rank 0 on GPU 0 and rank 1 on GPU 1");
    size_t bytes=number(argv[4],16,128*1024*1024),chunk=number(argv[5],16,128*1024*1024);
    int work=number(argv[6],0,4096);unsigned iters=number(argv[7],1,10000),warmup=number(argv[8],0,1000);
    std::string mode(argv[9]);
    if(bytes%16 || chunk%16 || (mode!="serial" && mode!="overlap")) throw std::invalid_argument("alignment/mode");
    if(preflight()!=0) return 2;
    return worker(rank,gpu,argv[3],bytes,chunk,work,iters,warmup,mode=="serial");
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n";return 1; }
}
