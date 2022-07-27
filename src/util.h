// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <cinttypes>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>

#include "timer.h"


/*
GAP Benchmark Suite
Author: Scott Beamer

Miscellaneous helpers that don't fit into classes
*/


static const int64_t kRandSeed = 27491095;


void PrintLabel(const std::string &label, const std::string &val) {
  printf("%-21s%7s\n", (label + ":").c_str(), val.c_str());
}

void PrintTime(const std::string &s, double seconds) {
  printf("%-21s%3.5lf\n", (s + ":").c_str(), seconds);
}

void PrintStep(const std::string &s, int64_t count) {
  printf("%-14s%14" PRId64 "\n", (s + ":").c_str(), count);
}

void PrintStep(int step, double seconds, int64_t count = -1) {
  if (count != -1)
    printf("%5d%11" PRId64 "  %10.5lf\n", step, count, seconds);
  else
    printf("%5d%23.5lf\n", step, seconds);
}

void PrintStep(const std::string &s, double seconds, int64_t count = -1) {
  if (count != -1)
    printf("%5s%11" PRId64 "  %10.5lf\n", s.c_str(), count, seconds);
  else
    printf("%5s%23.5lf\n", s.c_str(), seconds);
}

// Runs op and prints the time it took to execute labelled by label
#define TIME_PRINT(label, op) {   \
  Timer t_;                       \
  t_.Start();                     \
  (op);                           \
  t_.Stop();                      \
  PrintTime(label, t_.Seconds()); \
}


template <typename T_>
class RangeIter {
  T_ x_;
 public:
  explicit RangeIter(T_ x) : x_(x) {}
  bool operator!=(RangeIter const& other) const { return x_ != other.x_; }
  T_ const& operator*() const { return x_; }
  RangeIter& operator++() {
    ++x_;
    return *this;
  }
};

template <typename T_>
class Range{
  T_ from_;
  T_ to_;
 public:
  explicit Range(T_ to) : from_(0), to_(to) {}
  Range(T_ from, T_ to) : from_(from), to_(to) {}
  RangeIter<T_> begin() const { return RangeIter<T_>(from_); }
  RangeIter<T_> end() const { return RangeIter<T_>(to_); }
};

/**
 * @brief 物理地址转虚拟地址
 *
 * @param vaddr
 * @param paddr
 * @return 0 is success, -1 is failed!
 */
int mem_addr(unsigned long vaddr, unsigned long *paddr)
{
  int pageSize = getpagesize(); // //调用此函数获取系统设定的页面大小

  unsigned long v_pageIndex = vaddr / pageSize;            //计算此虚拟地址相对于0x0的经过的页面数
  unsigned long v_offset = v_pageIndex * sizeof(uint64_t); //计算在/proc/pid/page_map文件中的偏移量
  unsigned long page_offset = vaddr % pageSize;            //计算虚拟地址在页面中的偏移量
  uint64_t item = 0;                                       //存储对应项的值

  int fd = open("/proc/self/pagemap", O_RDONLY); //以只读方式打开/proc/pid/page_map
  if (fd == -1)                                  //判断是否打开失败

  {
    printf("open /proc/self/pagemap error\n");
    return -1;
  }

  if (lseek(fd, v_offset, SEEK_SET) == -1) //将游标移动到相应位置，即对应项的起始地址且判断是否移动失败

  {
    printf("sleek error\n");
    return -1;
  }

  if (read(fd, &item, sizeof(uint64_t)) != sizeof(uint64_t)) //读取对应项的值，并存入item中，且判断读取数据位数是否正确

  {
    printf("read item error\n");
    return -1;
  }

  if ((((uint64_t)1 << 63) & item) == 0) //判断present是否为0
  {
    printf("page present is 0\n");
    return -1;
  }

  uint64_t phy_pageIndex = (((uint64_t)1 << 55) - 1) & item; //计算物理页号，即取item的bit0-54

  *paddr = (phy_pageIndex * pageSize) + page_offset; //再加上页内偏移量就得到了物理地址
  return 0;
}
void print_address(const char *msg, unsigned long virt_addr)
{
  printf("[virtual address] %s : 0x%lx\n", msg, virt_addr);
  unsigned long phys_addr;
  int ret = mem_addr(virt_addr, &phys_addr);
  if (ret == 0)
  {
    printf("[physical address] %s : 0x%lx\n", msg, phys_addr);
  }
  
}

#endif  // UTIL_H_
