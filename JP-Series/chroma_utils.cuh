#pragma once
#include "ECLgraph.h"
#include "globals.cuh"

// GPU 指標容器
struct DevPtr {
    int *nidx_d{}, *nlist_d{}, *nlist2_d{};
    int *posscol_d{}, *posscol2_d{}, *color_d{}, *wl_d{};
    unsigned int *degree_list{}, *iteration_list_d{};
};

/* 空間配置 + 前置初始化 */
void allocAndInit(const ECLgraph& g, DevPtr& d);

/* 三步驟 kernel 封裝（需先 sl_allocate 完成 iteration_list） */
void ECL_GC_run(int blocks, const ECLgraph& g, DevPtr& d);

/* 驗證 & 統計輸出 */
void verifyAndPrintStats(const ECLgraph& g,
                         const int* color,
                         float runtime);
