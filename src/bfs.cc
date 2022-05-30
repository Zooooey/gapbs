// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

int64_t BUStep(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next) {
  int64_t awake_count = 0;
  next.reset();
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  //这里遍历整个graph？
  for (NodeID u=0; u < g.num_nodes(); u++) {
    //如果<0，说明没被遍历
    if (parent[u] < 0) {
      //printf("%p, R, NodeID\n",&u);
      //printf("%p, R, parent\n",&parent+u);
      //in_neigh是符合某种逻辑的邻居节点
      for (NodeID v : g.in_neigh(u)) {
        //printf("%p, R, NodeID\n",&v);
        if (front.get_bit(v)) {
          //完成前序节点的遍历？
          //ccy:读取前序节点，写入parent[u]
          parent[u] = v;
          //printf("%p, W, parent\n",&parent+u);
          awake_count++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
  return awake_count;
}


int64_t TDStep(const Graph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for reduction(+ : scout_count) nowait
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      //printf("%p, R, queue\n",&q_iter);
      //ccy:读取点id
      NodeID u = *q_iter;
      //printf("%p, R, NodeID\n",&u);
      //ccy:遍历该点所有的邻居，也就是读取这些邻居点。
      for (NodeID v : g.out_neigh(u)) {
        //printf("%p, R, NodeID\n",&v);
        //ccy:parent被读取，这里是为了获取邻居点v的出度
        NodeID curr_val = parent[v];
        //printf("%p, R, parent\n",parent[v]);
        if (curr_val < 0) {
          //修改parent，标明该点已经被遍历（未被遍历是负数，遍历后是该点的值？）
          if (compare_and_swap(parent[v], curr_val, u)) {
            //ccy:遍历该点完成，点被访问，推入局部队列。
            lqueue.push_back(v);
            //修改出度
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}

//遍历队列里的点，把这些点标记到位图里
void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    //printf("%p, R, queue\n",&q_iter);
    //ccy：读取点的id值
    NodeID u = *q_iter;
    //printf("%p, R, NodeID\n",&u);
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for nowait
    for (NodeID n=0; n < g.num_nodes(); n++){
      //printf("%p, R, NodeID\n",&n);
      if (bm.get_bit(n)){
        lqueue.push_back(n);
      }
    }
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++){
      parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
      //printf("%p, W, parent\n",&parent+n);
    }
  return parent;
}

pvector<NodeID> DOBFS(const Graph &g, NodeID source, int alpha = 15,
                      int beta = 18) {
  PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  printf("Graph : %p\n",&g);
  //初始化node数量的pvector
  pvector<NodeID> parent = InitParent(g);
  parent.printAddress();
  t.Stop();
  PrintStep("i", t.Seconds());
  //这个source就是bfs的起点，是随机传进来的。
  //printf("%p, R, NodeID\n",&source);
  parent[source] = source;
  //printf("%p, W, parent\n",&parent+source);
  //这个滑动队列
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.printAddress();
  //先把这个根节点push到queue里
  //printf("%p, R, NodeID\n",&source);
  queue.push_back(source);
  queue.slide_window();
  //分别建立了两个Bitmap
  Bitmap curr(g.num_nodes());
  curr.reset();
  curr.printAddress();
  Bitmap front(g.num_nodes());
  front.reset();
  front.printAddress();
  //获得图里边的数量
  int64_t edges_to_check = g.num_edges_directed();
  //获得根节点source的出度。
  int64_t scout_count = g.out_degree(source);
  while (!queue.empty()) {
    //点的出度大于某个阈值
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
      PrintStep("e", t.Seconds());
      awake_count = queue.size();
      //队列滑动，前面的点已经被标到位图里了
      queue.slide_window();
      do {
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStep(g, parent, front, curr);
        front.swap(curr);
        t.Stop();
        PrintStep("bu", t.Seconds(), awake_count);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueue(g, front, queue));
      PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {//点的出度小于某个阈值
      t.Start();
      //这个edges_to_check是汇总当前还有多少edges需要去check，减去scout_count是因为当前node的出度就是scout_count，后面要检查了，所以减去总数edges_to_check。
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue);
      queue.slide_window();
      t.Stop();
      PrintStep("td", t.Seconds(), queue.size());
    }
  }
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++){
    //printf("%p, R, parent\n",&parent+n);
    if (parent[n] < -1){
      //printf("%p, W, parent\n",&parent+n);
      parent[n] = -1;
    }
  }
  return parent;
}


void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  SourcePicker<Graph> sp(g, cli.start_vertex());
  auto BFSBound = [&sp] (const Graph &g) { return DOBFS(g, sp.PickNext()); };
  SourcePicker<Graph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}

