#pragma once
/* lingjing_explore — ARC-AGI-3 未知环境探索加速库（C ABI，供 Python ctypes 调用）
 *
 * 设计目标：
 * 1) 超快网格/状态哈希（替代反复 MD5 / 大 tuple 进 set）
 * 2) 高吞吐 visited（开放寻址），服务 BFS / beam / IDDFS
 * 3) beam top-k 选择（C++ 排序，Python 少做）
 * 4) 紧凑路径 BFS 调度器（路径存 int 动作码，避免 Python list 爆炸）
 *
 * 环境转移仍由 Python/Arcade 执行；C++ 管探索数据结构与调度。
 */

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#  ifdef LJ_EXPLORE_EXPORTS
#    define LJ_API __declspec(dllexport)
#  else
#    define LJ_API __declspec(dllimport)
#  endif
#else
#  define LJ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 哈希 ---- */
LJ_API uint64_t lj_hash_bytes(const uint8_t* data, size_t n);
LJ_API uint64_t lj_hash_grid_i8(const int8_t* data, int h, int w);
LJ_API uint64_t lj_hash_mix2(uint64_t a, uint64_t b);
LJ_API uint64_t lj_hash_mix3(uint64_t a, uint64_t b, uint64_t c);

/* ---- Visited 集合 ---- */
LJ_API void*   lj_visited_create(void);
LJ_API void    lj_visited_destroy(void* handle);
LJ_API int     lj_visited_insert(void* handle, uint64_t key);   /* 1=新插入, 0=已存在 */
LJ_API int     lj_visited_contains(void* handle, uint64_t key);
LJ_API size_t  lj_visited_size(void* handle);
LJ_API void    lj_visited_clear(void* handle);

/* ---- Beam：按 score 降序取 top-k 下标 ---- */
LJ_API void lj_beam_top_k(const double* scores, int n, int k, int* out_indices);

/* ---- 紧凑 BFS 前沿 ----
 * 动作编码：
 *   1..5,7 = 键盘/RESET
 *   点击 ACTION6： 0x6000000 | (x << 8) | y   （x,y 各 0..255）
 */
LJ_API void* lj_bfs_create(int max_nodes, int max_depth);
LJ_API void  lj_bfs_destroy(void* handle);
LJ_API int   lj_bfs_push_root(void* handle); /* 返回 node_id，失败 -1 */
LJ_API int   lj_bfs_pop(void* handle, int* out_node_id, int* out_depth);
LJ_API int   lj_bfs_path(void* handle, int node_id, int* out_actions, int max_out, int* out_len);
LJ_API int   lj_bfs_try_child(void* handle, int parent_id, int action_code,
                              uint64_t child_hash, int* out_child_id);
LJ_API void  lj_bfs_mark_seen(void* handle, uint64_t key); /* 标记根/已见状态 */
LJ_API size_t lj_bfs_queue_size(void* handle);
LJ_API size_t lj_bfs_seen_size(void* handle);

LJ_API const char* lj_explore_version(void);

#ifdef __cplusplus
}
#endif
