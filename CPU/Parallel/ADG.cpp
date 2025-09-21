/**********************************************************************
 * ADG_noCSV.cpp – Average-Degree Greedy PA  (CPU, OpenMP)
 * build : g++ -O3 -std=c++17 -fopenmp ADG_noCSV.cpp -o adg_cpu
 *********************************************************************/
 #include <algorithm>
 #include <sys/time.h>
 #include "ECLgraph.h"
  #include <bits/stdc++.h>
 #include <omp.h>
 #include "ECLgraph.h"
 static const int BPI = 32;  // bits per int
 static const int MSB = 1 << (BPI - 1);
 static const int Mask = (1 << (BPI / 2)) - 1;
 // source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
 static unsigned int hash(unsigned int val)
 {
   val = ((val >> 16) ^ val) * 0x45d9f3b;
   val = ((val >> 16) ^ val) * 0x45d9f3b;
   return (val >> 16) ^ val;
 }
 
 /* 16-bit 雜湊：只取低 16 bit 即可 */
 static inline uint16_t hash16(uint32_t x)
 {
     return hash(x) & 0xffffu;
 }
 static int init(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nlist2, int* const __restrict__ posscol, int* const __restrict__ posscol2, int* const __restrict__ color, int* const __restrict__ wl, const int threads,int* const __restrict__ priority)
 {
   int wlsize = 0;
   int maxrange = -1;
   #pragma omp parallel for num_threads(threads) default(none) reduction(max: maxrange) shared(nodes, wlsize, wl, nidx, nlist, nlist2, color, posscol,priority)
   for (int v = 0; v < nodes; v++) {
     int active;
     const int beg = nidx[v];
     const int end = nidx[v + 1];
     const int degv = end - beg;
     int v_priority=priority[v];
     const bool cond = (v_priority>>30);
     int pos = beg;
     if (cond) {
       int tmp;
       #pragma omp atomic capture
       tmp = wlsize++;
       wl[tmp] = v;
       for (int i = beg; i < end; i++) {
         const int nei = nlist[i];
         const int degn = nidx[nei + 1] - nidx[nei];
         const int priority_n=priority[nei];
         int hash_v = hash(v);
         int hash_nei = hash(nei);
 
         // if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
         if ((v_priority < priority_n) || ((v_priority == priority_n) && (hash_v < hash_nei)) || ((v_priority == priority_n) && (hash_v == hash_nei) && (v < nei))) {
             nlist2[pos] = nei;
           pos++;
         }
       }
     } else {
       active = 0;
       for (int i = beg; i < end; i++) {
         const int nei = nlist[i];
         const int degn = nidx[nei + 1] - nidx[nei];
         const int priority_n=priority[nei];
         int hash_v = hash(v);
         int hash_nei = hash(nei);
         if ((v_priority < priority_n) || ((v_priority == priority_n) && (hash_v < hash_nei)) || ((v_priority == priority_n) && (hash_v == hash_nei) && (v < nei))) {
         // if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
           active |= (unsigned int)MSB >> (i - beg);
           pos++;
         }
       }
     }
     const int range = pos - beg;
     maxrange = std::max(maxrange, range);  // reduction
     color[v] = (cond || (range == 0)) ? (range << (BPI / 2)) : active;
     posscol[v] = (range >= BPI) ? -1 : (MSB >> range);
   }
   if (maxrange >= Mask) {printf("too many active neighbors\n"); exit(-1);}
   #pragma omp parallel for num_threads(threads) default(none) shared(edges, posscol2)
   for (int i = 0; i < edges / BPI + 1; i++) posscol2[i] = -1;
   return wlsize;
 }
 // static int init(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nlist2, int* const __restrict__ posscol, int* const __restrict__ posscol2, int* const __restrict__ color, int* const __restrict__ wl, const int threads,int* const __restrict__ priority)
 // {
 //   int wlsize = 0;
 //   int maxrange = -1;
 //   #pragma omp parallel for num_threads(threads) default(none) reduction(max: maxrange) shared(nodes, wlsize, wl, nidx, nlist, nlist2, color, posscol,priority)
 //     for (int v = 0; v < nodes; v++) {
 //     int active;
 //     const int beg = nidx[v];
 //     const int end = nidx[v + 1];
 //     const int degv = end - beg;
 //     int v_priority=priority[v];
 //     const bool cond = (v_priority>>30);
 //     if (degv>=BPI&&cond==0)
 //     printf("v_priority[%d]:%d=%d=%d\n",v,v_priority,v_priority>>30,degv);
 //     // const bool cond = (degv >= BPI);
 //     int pos = beg;
 //     if (cond) {
 //       int tmp;
 //       #pragma omp atomic capture
 //       tmp = wlsize++;
 //       wl[tmp] = v;
 //       for (int i = beg; i < end; i++) {
 //         const int nei = nlist[i];
 //         const int degn = nidx[nei + 1] - nidx[nei];
 //         if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
 //           nlist2[pos] = nei;
 //           pos++;
 //         }
 //       }
 //     } else {
 //       active = 0;
 //       for (int i = beg; i < end; i++) {
 //         const int nei = nlist[i];
 //         const int degn = nidx[nei + 1] - nidx[nei];
 //         if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
 //           active |= (unsigned int)MSB >> (i - beg);
 //           pos++;
 //         }
 //       }
 //     }
 //     const int range = pos - beg;
 //     maxrange = std::max(maxrange, range);  // reduction
 //     color[v] = (cond || (range == 0)) ? (range << (BPI / 2)) : active;
 //     posscol[v] = (range >= BPI) ? -1 : (MSB >> range);
 //   }
 //   if (maxrange >= Mask) {printf("too many active neighbors\n"); exit(-1);}
 //   #pragma omp parallel for num_threads(threads) default(none) shared(edges, posscol2)
 //   for (int i = 0; i < edges / BPI + 1; i++) posscol2[i] = -1;
 //   return wlsize;
 // }
 
 
 void runLarge(const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ posscol, volatile int* const __restrict__ posscol2, volatile int* const __restrict__ color, const int* const __restrict__ wl, const int wlsize, const int threads)
 {
   if (wlsize != 0) {
     bool again;
     #pragma omp parallel num_threads(threads) default(none) shared(wlsize, wl, nidx, nlist, color, posscol, posscol2) private(again)
     do {
       again = false;
       #pragma omp for nowait
       for (int w = 0; w < wlsize; w++) {
         bool shortcut = true;
         bool done = true;
         const int v = wl[w];
         int data;  // const
         #pragma omp atomic read
         data = color[v];
         const int range = data >> (BPI / 2);
         if (range > 0) {
           const int beg = nidx[v];
           int pcol = posscol[v];
           const int mincol = data & Mask;
           const int maxcol = mincol + range;
           const int end = beg + maxcol;
           const int offs = beg / BPI;
           for (int i = beg; i < end; i++) {
             const int nei = nlist[i];
             int neidata;  // const
             #pragma omp atomic read
             neidata = color[nei];
             const int neirange = neidata >> (BPI / 2);
             if (neirange == 0) {
               const int neicol = neidata;
               if (neicol < BPI) {
                 pcol &= ~((unsigned int)MSB >> neicol);
               } else {
                 if ((mincol <= neicol) && (neicol < maxcol)) {
                   int pc;  // const
                   #pragma omp atomic read
                   pc = posscol2[offs + neicol / BPI];
                   if ((pc << (neicol % BPI)) < 0) {
                     #pragma omp atomic update
                     posscol2[offs + neicol / BPI] &= ~((unsigned int)MSB >> (neicol % BPI));
                   }
                 }
               }
             } else {
               done = false;
               const int neimincol = neidata & Mask;
               const int neimaxcol = neimincol + neirange;
               if ((neimincol <= mincol) && (neimaxcol >= mincol)) shortcut = false;
             }
           }
           int val = pcol;
           int mc = 0;
           if (pcol == 0) {
             const int offs = beg / BPI;
             mc = std::max(1, mincol / BPI) - 1;
             do {
               mc++;
               #pragma omp atomic read
               val = posscol2[offs + mc];
             } while (val == 0);
           }
           int newmincol = mc * BPI + __builtin_clz(val);
           if (mincol != newmincol) shortcut = false;
           if (shortcut || done) {
             pcol = (newmincol < BPI) ? ((unsigned int)MSB >> newmincol) : 0;
           } else {
             const int maxcol = mincol + range;
             const int range = maxcol - newmincol;
             newmincol = (range << (BPI / 2)) | newmincol;
             again = true;
           }
           posscol[v] = pcol;
           #pragma omp atomic write
           color[v] = newmincol;
         }
       }
     } while (again);
   }
 }
 
 
 void runSmall(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, volatile int* const __restrict__ posscol, int* const __restrict__ color, const int threads)
 {
   bool again;
   #pragma omp parallel num_threads(threads) default(none) shared(nodes, nidx, nlist, color, posscol) private(again)
   do {
     again = false;
     #pragma omp for nowait
     for (int v = 0; v < nodes; v++) {
       int pcol;
       #pragma omp atomic read
       pcol = posscol[v];
       if (__builtin_popcount(pcol) > 1) {
         const int beg = nidx[v];
         int active = color[v];
         int allnei = 0;
         int keep = active;
         do {
           const int old = active;
           active &= active - 1;
           const int curr = old ^ active;
           const int i = beg + __builtin_clz(curr);
           const int nei = nlist[i];
           int neipcol;  // const
           #pragma omp atomic read
           neipcol = posscol[nei];
           allnei |= neipcol;
           if ((pcol & neipcol) == 0) {
             pcol &= pcol - 1;
             keep ^= curr;
           } else if (__builtin_popcount(neipcol) == 1) {
             pcol ^= neipcol;
             keep ^= curr;
           }
         } while (active != 0);
         if (keep != 0) {
           const int best = (unsigned int)MSB >> __builtin_clz(pcol);
           if ((best & ~allnei) != 0) {
             pcol = best;
             keep = 0;
           }
         }
         again |= keep;
         if (keep == 0) keep = __builtin_clz(pcol);
         color[v] = keep;
         #pragma omp atomic write
         posscol[v] = pcol;
       }
     }
   } while (again);
 }
 
 
 struct CPUTimer
 {
   timeval beg, end;
   void start() {gettimeofday(&beg, NULL);}
   float stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) * 0.000001f;}
 };
 
 
 /********************************************************************
  * compute_ADG  –  Average-Degree Greedy Priority Allocation
  *                (thread-local bucket, ZERO atomic)
  *
  * priority[v] = (iter << 16) | (deg & 0xffff)
  * eps          – 建議 0.0 ~ 0.1；0 表嚴格「≤ avg」，越大越快但色數可能多 1~2。
  ********************************************************************/
 void compute_ADG(const ECLgraph& g,
     int threads,
     std::vector<int>& priority,
     double eps = 0.0)
 {
 const int N = g.nodes;
 
 /* 初始度數與活躍集合 ----------------------------------------- */
 std::vector<int>  deg(N);
 std::vector<char> active(N, 1);
 
 #pragma omp parallel for num_threads(threads) schedule(static)
 for (int v = 0; v < N; ++v)
 deg[v] = g.nindex[v + 1] - g.nindex[v];
 
 long long sum = 0;
 int        cnt = 0;
 int        iter = 0;
 
 /* ---------------- parallel region ---------------------------- */
 #pragma omp parallel num_threads(threads) shared(deg,active,priority,iter,sum,cnt)
 {
 std::vector<int> bucket;          // thread-local 鄰居遞減暫存
 bucket.reserve(256);
 
 while (true) {
 /* (1) 重新計算 Σdeg 與活躍數量 |U| */
 #pragma omp single
 { sum = 0; cnt = 0; }
 
 #pragma omp for reduction(+:sum,cnt) schedule(static)
 for (int v = 0; v < N; ++v)
    if (active[v]) { sum += deg[v]; ++cnt; }
 
 double avg;
 #pragma omp single
 { avg = cnt ? double(sum) / cnt : 0.0; }
 
 if (cnt == 0) break;          // 全部剝完
 
 const double threshold = (1.0 + eps) * avg;
 
 /* (2) 剝皮 + 收集鄰居到 bucket（無 atomic） */
 bucket.clear();
 int local_removed = 0;        // thread-local
 
 #pragma omp for schedule(dynamic,256) nowait
 for (int v = 0; v < N; ++v) {
    if (!active[v] || deg[v] > threshold) continue;
 
    active[v] = 0;
    int deg_v = (g.nindex[v + 1] - g.nindex[v])>=BPI;
 
    priority[v] = ((deg_v) << 30) |((iter + 1) << 16) | (deg[v] & 0xffff);
    ++local_removed;
 
    for (int idx = g.nindex[v]; idx < g.nindex[v + 1]; ++idx) {
        int nei = g.nlist[idx];
        if (active[nei]) bucket.push_back(nei);
    }
 }
 
 /* (3) barrier 後，批次把 bucket 裡的鄰居度數 –1 */
 #pragma omp barrier
 #pragma omp for schedule(static,1024)
 for (size_t i = 0; i < bucket.size(); ++i)
    --deg[bucket[i]];
 
 /* (4) 下一回合 */
 #pragma omp single
 { ++iter; }
 #pragma omp barrier
 } /* while */
 }     /* end parallel */
 
 /* (5) 補上最後仍 active 頂點的 priority（極少數） */
 #pragma omp parallel for num_threads(threads) schedule(static)
 for (int v = 0; v < N; ++v)
 if (priority[v] == 0){
     int deg_v = (g.nindex[v + 1] - g.nindex[v])>=BPI;
     priority[v] = ((deg_v) << 30) |((iter + 1) << 16) | (deg[v] & 0xffff); 
 }
 
 printf("ADG finished: iterations = %d\n", iter);
 }
 
 int main(int argc, char** argv)
 {
   if (argc != 3) {
     fprintf(stderr, "USAGE: %s <graph.egr> <threads>\n", argv[0]);
     return 0;
   }
   int threads = atoi(argv[2]);
   if (threads < 1) { fprintf(stderr, "threads must be >= 1\n"); return 0; }
 
   ECLgraph g = readECLgraph(argv[1]);
   printf("nodes=%d edges=%d avgDeg=%.2f\n", g.nodes, g.edges, 1.0 * g.edges / g.nodes);
 
   std::vector<int> priority(g.nodes, 0);
 
     int* const color = new int [g.nodes];
     int* const nlist2 = new int [g.edges];
     int* const posscol = new int [g.nodes];
     int* const posscol2 = new int [g.edges / BPI + 1];
     int* const wl = new int [g.nodes];
   
     CPUTimer timer;
     timer.start();
     compute_ADG(g, threads, priority,0);
 
     printf("Start init \n");
 
     const int wlsize = init(g.nodes, g.edges, g.nindex, g.nlist, nlist2, posscol, posscol2, color, wl, threads,priority.data());
     printf("Start runLarge \n");
 
     runLarge(g.nindex, nlist2, posscol, posscol2, color, wl, wlsize, threads);
     printf("Start runSmall \n");
 
     runSmall(g.nodes, g.nindex, g.nlist, posscol, color, threads);
     const float runtime = timer.stop();
   
     printf("runtime:    %.6f ms\n", runtime*1000);
     printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
     printf("throughput: %.6f Medges/s\n", g.edges * 0.000001 / runtime);
   
     for (int v = 0; v < g.nodes; v++) {
       if (color[v] < 0) {printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n", v, g.nindex[v + 1] - g.nindex[v]);  exit(-1);}
       for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
         if (color[g.nlist[i]] == color[v]) {printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n", color[v], v, g.nlist[i]);  exit(-1);}
       }
     }
     printf("result verification passed\n");
   
     const int vals = 16;
     int c[vals];
     for (int i = 0; i < vals; i++) c[i] = 0;
     int cols = -1;
     for (int v = 0; v < g.nodes; v++) {
       cols = std::max(cols, color[v]);
       if (color[v] < vals) c[color[v]]++;
     }
     cols++;
     printf("colors used: %d\n", cols);
   
     int sum = 0;
     for (int i = 0; i < std::min(vals, cols); i++) {
       sum += c[i];
       printf("col %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / g.nodes);
     }
   
     delete [] color;
     delete [] nlist2;
     delete [] posscol;
     delete [] posscol2;
     delete [] wl;
     freeECLgraph(g);
     return 0;
   }