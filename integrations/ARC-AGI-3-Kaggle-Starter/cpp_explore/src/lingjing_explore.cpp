#include "lingjing_explore.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <vector>

/* ---------- FNV-1a 64：极快、足够做去重主键 ---------- */
static inline uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

extern "C" uint64_t lj_hash_bytes(const uint8_t* data, size_t n) {
    if (!data || n == 0) return 0;
    return fnv1a64(data, n);
}

extern "C" uint64_t lj_hash_grid_i8(const int8_t* data, int h, int w) {
    if (!data || h <= 0 || w <= 0) return 0;
    return fnv1a64(reinterpret_cast<const uint8_t*>(data),
                   static_cast<size_t>(h) * static_cast<size_t>(w));
}

extern "C" uint64_t lj_hash_mix2(uint64_t a, uint64_t b) {
    // splitmix 风格混合，降低 (level,score,grid) 拼接碰撞
    uint64_t x = a + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    x ^= b + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x;
}

extern "C" uint64_t lj_hash_mix3(uint64_t a, uint64_t b, uint64_t c) {
    return lj_hash_mix2(lj_hash_mix2(a, b), c);
}

/* ---------- Visited ---------- */
struct Visited {
    std::unordered_set<uint64_t> s;
};

extern "C" void* lj_visited_create(void) { return new Visited(); }
extern "C" void lj_visited_destroy(void* handle) { delete static_cast<Visited*>(handle); }

extern "C" int lj_visited_insert(void* handle, uint64_t key) {
    auto* v = static_cast<Visited*>(handle);
    auto r = v->s.insert(key);
    return r.second ? 1 : 0;
}

extern "C" int lj_visited_contains(void* handle, uint64_t key) {
    auto* v = static_cast<Visited*>(handle);
    return v->s.find(key) != v->s.end() ? 1 : 0;
}

extern "C" size_t lj_visited_size(void* handle) {
    return static_cast<Visited*>(handle)->s.size();
}

extern "C" void lj_visited_clear(void* handle) {
    static_cast<Visited*>(handle)->s.clear();
}

/* ---------- Beam top-k ---------- */
extern "C" void lj_beam_top_k(const double* scores, int n, int k, int* out_indices) {
    if (!scores || !out_indices || n <= 0 || k <= 0) return;
    if (k > n) k = n;
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    for (int i = 0; i < k; ++i) out_indices[i] = idx[i];
}

/* ---------- Compact BFS ---------- */
struct BfsNode {
    int parent = -1;
    int action = 0;
    int depth = 0;
};

struct Bfs {
    int max_nodes = 0;
    int max_depth = 0;
    std::vector<BfsNode> nodes;
    std::queue<int> q;
    std::unordered_set<uint64_t> seen;
};

extern "C" void* lj_bfs_create(int max_nodes, int max_depth) {
    auto* b = new Bfs();
    b->max_nodes = max_nodes > 0 ? max_nodes : 20000;
    b->max_depth = max_depth > 0 ? max_depth : 40;
    b->nodes.reserve(b->max_nodes);
    return b;
}

extern "C" void lj_bfs_destroy(void* handle) { delete static_cast<Bfs*>(handle); }

extern "C" int lj_bfs_push_root(void* handle) {
    auto* b = static_cast<Bfs*>(handle);
    b->nodes.clear();
    while (!b->q.empty()) b->q.pop();
    b->seen.clear();
    BfsNode root;
    root.parent = -1;
    root.action = 0;
    root.depth = 0;
    b->nodes.push_back(root);
    b->q.push(0);
    return 0;
}

extern "C" int lj_bfs_pop(void* handle, int* out_node_id, int* out_depth) {
    auto* b = static_cast<Bfs*>(handle);
    if (b->q.empty()) return 0;
    int id = b->q.front();
    b->q.pop();
    if (out_node_id) *out_node_id = id;
    if (out_depth) *out_depth = b->nodes[id].depth;
    return 1;
}

extern "C" int lj_bfs_path(void* handle, int node_id, int* out_actions, int max_out, int* out_len) {
    auto* b = static_cast<Bfs*>(handle);
    if (!out_actions || !out_len || node_id < 0 || node_id >= (int)b->nodes.size()) return 0;
    std::vector<int> rev;
    for (int id = node_id; id > 0; id = b->nodes[id].parent) {
        rev.push_back(b->nodes[id].action);
    }
    std::reverse(rev.begin(), rev.end());
    int n = (int)rev.size();
    if (n > max_out) n = max_out;
    for (int i = 0; i < n; ++i) out_actions[i] = rev[i];
    *out_len = n;
    return 1;
}

extern "C" int lj_bfs_try_child(void* handle, int parent_id, int action_code,
                                uint64_t child_hash, int* out_child_id) {
    auto* b = static_cast<Bfs*>(handle);
    if (parent_id < 0 || parent_id >= (int)b->nodes.size()) return 0;
    if ((int)b->nodes.size() >= b->max_nodes) return 0;
    int nd = b->nodes[parent_id].depth + 1;
    if (nd > b->max_depth) return 0;
    if (!b->seen.insert(child_hash).second) return 0; // 已见
    BfsNode ch;
    ch.parent = parent_id;
    ch.action = action_code;
    ch.depth = nd;
    int id = (int)b->nodes.size();
    b->nodes.push_back(ch);
    b->q.push(id);
    if (out_child_id) *out_child_id = id;
    return 1;
}

extern "C" void lj_bfs_mark_seen(void* handle, uint64_t key) {
    static_cast<Bfs*>(handle)->seen.insert(key);
}

extern "C" size_t lj_bfs_queue_size(void* handle) {
    return static_cast<Bfs*>(handle)->q.size();
}

extern "C" size_t lj_bfs_seen_size(void* handle) {
    return static_cast<Bfs*>(handle)->seen.size();
}

extern "C" const char* lj_explore_version(void) {
    return "lingjing_explore/1.0.0-arcagi3";
}
