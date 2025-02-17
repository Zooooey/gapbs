// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef SLIDING_QUEUE_H_
#define SLIDING_QUEUE_H_

#include <algorithm>

#include "platform_atomics.h"
#include "util.h"


/*
GAP Benchmark Suite
Class:  SlidingQueue
Author: Scott Beamer

Double-buffered queue so appends aren't seen until SlideWindow() called
 - Use QueueBuffer when used in parallel to avoid false sharing by doing
   bulk appends from thread-local storage
*/


template <typename T>
class QueueBuffer;

template <typename T>
class SlidingQueue {
  T *shared;
  size_t shared_in;
  size_t shared_out_start;
  size_t shared_out_end;
  size_t siz;
  friend class QueueBuffer<T>;

 public:
  explicit SlidingQueue(size_t shared_size) {
    shared = new T[shared_size];
    siz = shared_size;
    reset();
  }

  void printAddress(){
    print_address("  queue start", (unsigned long)(void*)shared);
    print_address("  queue end", (unsigned long)(void*)shared+siz);
    // printf("  queue start: %p\n",&shared);
    // printf("  queue end: %p\n",&shared+siz);
  }

  ~SlidingQueue() {
    delete[] shared;
  }

  void push_back(T to_add) {
    //printf("%p, W, queue\n",&shared[shared_in]);
    shared[shared_in++] = to_add;
  }

  bool empty() const {
    return shared_out_start == shared_out_end;
  }

  void reset() {
    shared_out_start = 0;
    shared_out_end = 0;
    shared_in = 0;
  }

  void slide_window() {
    shared_out_start = shared_out_end;
    shared_out_end = shared_in;
  }

  typedef T* iterator;

  iterator begin() const {
    return shared + shared_out_start;
  }

  iterator end() const {
    return shared + shared_out_end;
  }

  size_t size() const {
    return end() - begin();
  }
};


//QueueBuffer是一个局部队列，当队列满的时候，flush操作把局部队列里的内容拷贝给全局队列，并清空局部队列。
template <typename T>
class QueueBuffer {
  size_t in;
  T *local_queue;
  SlidingQueue<T> &sq;
  const size_t local_size;

 public:
  explicit QueueBuffer(SlidingQueue<T> &master, size_t given_size = 16384)
      : sq(master), local_size(given_size) {
    in = 0;
    local_queue = new T[local_size];
  }

  ~QueueBuffer() {
    delete[] local_queue;
  }

  void push_back(T to_add) {
    if (in == local_size)
      flush();
    //printf("%p, W, queue\n",&local_queue[in]);
    local_queue[in++] = to_add;
  }

  void flush() {
    //把当前local queue了的内容追加给全局的queue
    T *shared_queue = sq.shared;
    size_t copy_start = fetch_and_add(sq.shared_in, in);
    std::copy(local_queue, local_queue+in, shared_queue+copy_start);
    in = 0;
  }
};

#endif  // SLIDING_QUEUE_H_
