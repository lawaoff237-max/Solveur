// =============================================================================
//  kangaroo.cu  -  Pollard Kangaroo (lambda) multi-clés / GPU+CPU
//  Courbe secp256k1, intervalle borné [a, b].
//  Portage CUDA de kangaroo-9.cpp : la table des points distingués (DP),
//  la détection de collisions, les relations multi-clés, la sauvegarde/
//  reprise et le CLI restent sur CPU (inchangés, déjà multi-thread).
//  La marche du kangourou (le calcul lourd) tourne sur GPU (un thread CUDA
//  = un kangourou) ET continue de tourner sur CPU (threads std::thread),
//  pour utiliser TOUT le matériel disponible (tous les GPU + tous les
//  cœurs CPU restants).
//
//  Compilation (Colab / Kaggle, détecte l'architecture du GPU attribué) :
//     nvcc -O3 -std=c++17 -Xptxas -O3 -Xcompiler -O3,-pthread,-funroll-loops -arch=native kangaroo.cu -o kangaroo
//  Si "-arch=native" n'est pas supporté par votre nvcc, remplacez par par exemple :
//     nvcc -O3 -std=c++17 -Xptxas -O3 -Xcompiler -O3,-pthread,-funroll-loops \
//          -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
//          -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
//          kangaroo.cu -o kangaroo
//  -Xptxas -O3 force l'optimisation du code GPU (ptxas) ; -funroll-loops aide la partie CPU
//  (cpu_worker), comme pour kangaroo-9.cpp.
//
//  Utilisation : identique au programme original + options GPU :
//     -gpu 0|1        active/désactive le GPU (défaut : auto, actif si un GPU est détecté)
//     -gb <n>         nombre de blocks CUDA par GPU (défaut : 256)
//     -gt <n>         threads par block CUDA (défaut : 256)   -> kangourous GPU = gb*gt*nGPU
//     -gs <n>         pas (jumps) par appel de noyau avant de relever les DP (défaut : 48)
//     -c <n>          threads CPU additionnels (défaut : cœurs - 1 si GPU, sinon tous les cœurs)
//
//  NOTE IMPORTANTE : ce fichier doit être compilé et testé sur une machine
//  avec un GPU CUDA (Colab/Kaggle) ; il n'a pas pu être compilé avec nvcc
//  dans l'environnement où il a été écrit (pas de GPU/nvcc disponible ici).
//  Seule la partie arithmétique pure a pu être vérifiée en syntaxe avec g++
//  (voir mode CUDA_STUB ci-dessous). Testez `-test` avant un usage réel.
// =============================================================================

#if !defined(__CUDACC__)
#define __host__
#define __device__
#define __forceinline__ inline
#define CUDA_STUB 1
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <csignal>
#include <ctime>
#include <fstream>
#include <sstream>

#ifndef CUDA_STUB
#include <cuda_runtime.h>
#endif

using namespace std;
typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;
typedef __int128 i128;

// ============================ entiers 256 bits ===============================
struct U256 { u64 d[4]; };
typedef U256 fe;

static const u64 P0 = 0xFFFFFFFEFFFFFC2FULL;
static const u64 CP = 0x1000003D1ULL;            // 2^256 mod p

__host__ __device__ __forceinline__ bool u_zero(const U256& a) { return !(a.d[0] | a.d[1] | a.d[2] | a.d[3]); }
__host__ __device__ __forceinline__ bool u_eq(const U256& a, const U256& b) {
    return a.d[0] == b.d[0] && a.d[1] == b.d[1] && a.d[2] == b.d[2] && a.d[3] == b.d[3];
}
__host__ __device__ __forceinline__ int u_cmp(const U256& a, const U256& b) {
    for (int i = 3; i >= 0; i--) { if (a.d[i] != b.d[i]) return a.d[i] > b.d[i] ? 1 : -1; }
    return 0;
}
__host__ __device__ __forceinline__ U256 u_from(u128 v) { U256 r; r.d[0] = (u64)v; r.d[1] = (u64)(v >> 64); r.d[2] = r.d[3] = 0; return r; }
__host__ __device__ __forceinline__ U256 u_add(const U256& a, const U256& b) {
    U256 r; u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)a.d[i] + b.d[i]; r.d[i] = (u64)c; c >>= 64; }
    return r;
}
__host__ __device__ __forceinline__ U256 u_sub(const U256& a, const U256& b) {
    U256 r; u64 br = 0;
    for (int i = 0; i < 4; i++) { u128 t = (u128)a.d[i] - b.d[i] - br; r.d[i] = (u64)t; br = (u64)(t >> 64) & 1; }
    return r;
}
__host__ __device__ __forceinline__ int u_bits(const U256& a) {
#ifdef __CUDA_ARCH__
    for (int i = 3; i >= 0; i--) if (a.d[i]) return i * 64 + 64 - __clzll((long long)a.d[i]);
    return 0;
#else
    for (int i = 3; i >= 0; i--) if (a.d[i]) return i * 64 + 64 - __builtin_clzll(a.d[i]);
    return 0;
#endif
}
__host__ __device__ __forceinline__ void u_shr1(U256& a) {
    for (int i = 0; i < 3; i++) a.d[i] = (a.d[i] >> 1) | (a.d[i + 1] << 63);
    a.d[3] >>= 1;
}
static bool hex2u(string s, U256& r) {
    if (s.size() > 1 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    if (s.empty() || s.size() > 64) return false;
    memset(&r, 0, sizeof(r));
    int pos = 0;
    for (int i = (int)s.size() - 1; i >= 0; i--, pos++) {
        char c = s[i]; int v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        else return false;
        r.d[pos / 16] |= (u64)v << ((pos % 16) * 4);
    }
    return true;
}
static string u2hex(const U256& a) {
    char b[80]; snprintf(b, sizeof b, "%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3], (unsigned long long)a.d[2], (unsigned long long)a.d[1], (unsigned long long)a.d[0]);
    return string(b);
}

// ================================ corps Fp ===================================
__host__ __device__ __forceinline__ void add_small(U256& r, u64 v) {
    u128 c = (u128)r.d[0] + v; r.d[0] = (u64)c; c >>= 64;
    for (int i = 1; i < 4 && c; i++) { c += r.d[i]; r.d[i] = (u64)c; c >>= 64; }
}
__host__ __device__ __forceinline__ void sub_small(U256& r, u64 v) {
    u64 br = r.d[0] < v; r.d[0] -= v;
    for (int i = 1; i < 4 && br; i++) { br = (r.d[i] == 0); r.d[i]--; }
}
__host__ __device__ __forceinline__ void fe_norm(fe& r) {
    if (r.d[3] == ~0ULL && r.d[2] == ~0ULL && r.d[1] == ~0ULL && r.d[0] >= P0) add_small(r, CP);
}
__host__ __device__ __forceinline__ void fe_add(fe& r, const fe& a, const fe& b) {
    u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)a.d[i] + b.d[i]; r.d[i] = (u64)c; c >>= 64; }
    if (c) add_small(r, CP);
    fe_norm(r);
}
__host__ __device__ __forceinline__ void fe_sub(fe& r, const fe& a, const fe& b) {
    u64 br = 0;
    for (int i = 0; i < 4; i++) { u128 t = (u128)a.d[i] - b.d[i] - br; r.d[i] = (u64)t; br = (u64)(t >> 64) & 1; }
    if (br) sub_small(r, CP);
}
__host__ __device__ __forceinline__ void fe_neg(fe& r, const fe& a) {
    if (u_zero(a)) { r = a; return; }
    fe z; z.d[0] = z.d[1] = z.d[2] = z.d[3] = 0; fe_sub(r, z, a);
}
// Reduction commune (2^256 mod p = CP) : prend un produit brut de 8 mots 64 bits et le ramene < p.
// Partagee par fe_mul et fe_sqr pour eviter la duplication (host ET device).
__host__ __device__ __forceinline__ void fe_reduce_from_t(fe& r, u64 t[8]) {
    u64 o[4]; u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)t[4 + i] * CP + t[i]; o[i] = (u64)c; c >>= 64; }
    u64 top = (u64)c;
    c = (u128)top * CP;
    for (int i = 0; i < 4; i++) { c += o[i]; o[i] = (u64)c; c >>= 64; }
    r.d[0]=o[0]; r.d[1]=o[1]; r.d[2]=o[2]; r.d[3]=o[3];
    if (c) add_small(r, CP);
    fe_norm(r);
}
__host__ __device__ __forceinline__ void fe_mul(fe& r, const fe& a, const fe& b) {
    u64 t[8]; u128 c = 0;
    for (int j = 0; j < 4; j++) { c += (u128)a.d[0] * b.d[j]; t[j] = (u64)c; c >>= 64; }
    t[4] = (u64)c;
    for (int i = 1; i < 4; i++) {
        c = 0;
        for (int j = 0; j < 4; j++) { c += (u128)a.d[i] * b.d[j] + t[i + j]; t[i + j] = (u64)c; c >>= 64; }
        t[i + 4] = (u64)c;
    }
    fe_reduce_from_t(r, t);
}
// Squaring dediee : exploite a_i*a_j = a_j*a_i pour ne faire que 10 multiplications 64x64
// (6 termes croises + 4 carres) au lieu des 16 de fe_mul(a,a,a) -> ~1.5x plus rapide.
// Verifiee par fuzzing cote host (2M tirages + cas limites) contre l'ancienne fe_mul(a,a) ;
// code purement entier (u64/u128), identique sur device. C'est fe_inv() qui en beneficie le
// plus : sur GPU elle est appelee a CHAQUE saut (pas de lot Montgomery comme sur CPU), et
// elle est faite a 255/270 de squarings -> l'essentiel du cout d'un pas GPU vient de la.
__host__ __device__ __forceinline__ void fe_sqr(fe& r, const fe& a) {
    u64 OD[7] = {0,0,0,0,0,0,0};   // somme des termes croises a_i*a_j (i<j), sans le facteur 2
    u128 c;
    c = 0;
    for (int j = 1; j < 4; j++) { c += (u128)a.d[0] * a.d[j]; OD[j] = (u64)c; c >>= 64; }
    OD[4] = (u64)c;
    c = 0;
    for (int j = 2; j < 4; j++) { c += (u128)a.d[1] * a.d[j] + OD[1 + j]; OD[1 + j] = (u64)c; c >>= 64; }
    OD[5] = (u64)c;
    c = (u128)a.d[2] * a.d[3] + OD[5];
    OD[5] = (u64)c; c >>= 64;
    OD[6] = (u64)c;
    u64 t[8]; u64 carry = 0;                      // double OD (decalage 1 bit) -> t[0..7]
    for (int k = 0; k < 7; k++) { u64 v = OD[k]; t[k] = (v << 1) | carry; carry = v >> 63; }
    t[7] = carry;
    u64 dcarry = 0;                               // ajoute les carres a_i^2 sur des positions non chevauchantes
    for (int i = 0; i < 4; i++) {
        u128 sq = (u128)a.d[i] * a.d[i];
        u64 lo = (u64)sq, hi = (u64)(sq >> 64);
        u128 s = (u128)t[2 * i] + lo + dcarry;
        t[2 * i] = (u64)s; dcarry = (u64)(s >> 64);
        s = (u128)t[2 * i + 1] + hi + dcarry;
        t[2 * i + 1] = (u64)s; dcarry = (u64)(s >> 64);
    }
    fe_reduce_from_t(r, t);
}
__host__ __device__ __forceinline__ void fe_sqrn(fe& r, const fe& a, int n) { r = a; for (int i = 0; i < n; i++) fe_sqr(r, r); }

__host__ __device__ inline void fe_inv(fe& r, const fe& a) {   // a^(p-2), chaîne d'addition de libsecp256k1
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    fe_sqr(x2, a);       fe_mul(x2, x2, a);
    fe_sqr(x3, x2);      fe_mul(x3, x3, a);
    fe_sqrn(x6, x3, 3);  fe_mul(x6, x6, x3);
    fe_sqrn(x9, x6, 3);  fe_mul(x9, x9, x3);
    fe_sqrn(x11, x9, 2); fe_mul(x11, x11, x2);
    fe_sqrn(x22, x11, 11);   fe_mul(x22, x22, x11);
    fe_sqrn(x44, x22, 22);   fe_mul(x44, x44, x22);
    fe_sqrn(x88, x44, 44);   fe_mul(x88, x88, x44);
    fe_sqrn(x176, x88, 88);  fe_mul(x176, x176, x88);
    fe_sqrn(x220, x176, 44); fe_mul(x220, x220, x44);
    fe_sqrn(x223, x220, 3);  fe_mul(x223, x223, x3);
    fe_sqrn(t, x223, 23); fe_mul(t, t, x22);
    fe_sqrn(t, t, 5);     fe_mul(t, t, a);
    fe_sqrn(t, t, 3);     fe_mul(t, t, x2);
    fe_sqrn(t, t, 2);     fe_mul(r, t, a);
}
__host__ __device__ inline void fe_pow(fe& r, const fe& a, const U256& e) {
    fe res; res.d[0]=1; res.d[1]=res.d[2]=res.d[3]=0;
    int nb = u_bits(e);
    for (int i = nb - 1; i >= 0; i--) {
        fe_sqr(res, res);
        if ((e.d[i / 64] >> (i % 64)) & 1) fe_mul(res, res, a);
    }
    r = res;
}

// ================================ courbe =====================================
struct Pt { fe x, y; bool inf; };
struct Jac { fe X, Y, Z; };

static U256 SECP_N, GX, GY;
static Pt GEN;

__host__ __device__ __forceinline__ Jac jac_inf() { Jac r; memset(&r, 0, sizeof r); r.X.d[0] = 1; r.Y.d[0] = 1; return r; }
__host__ __device__ __forceinline__ Jac jac_from(const Pt& p) {
    if (p.inf) return jac_inf();
    Jac r; r.X = p.x; r.Y = p.y; r.Z.d[0]=1; r.Z.d[1]=r.Z.d[2]=r.Z.d[3]=0; return r;
}
__host__ __device__ inline void jac_dbl(Jac& r, const Jac& a) {
    if (u_zero(a.Z) || u_zero(a.Y)) { r = jac_inf(); return; }
    fe A, B, C, D, E, F, t, X3, Y3, Z3, C8;
    fe_sqr(A, a.X); fe_sqr(B, a.Y); fe_sqr(C, B);
    fe_add(t, a.X, B); fe_sqr(t, t); fe_sub(t, t, A); fe_sub(t, t, C); fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_sub(X3, F, D); fe_sub(X3, X3, D);
    fe_sub(t, D, X3); fe_mul(Y3, E, t);
    fe_add(C8, C, C); fe_add(C8, C8, C8); fe_add(C8, C8, C8);
    fe_sub(Y3, Y3, C8);
    fe_mul(Z3, a.Y, a.Z); fe_add(Z3, Z3, Z3);
    r.X = X3; r.Y = Y3; r.Z = Z3;
}
__host__ __device__ inline void jac_add(Jac& r, const Jac& a, const Jac& b) {
    if (u_zero(a.Z)) { r = b; return; }
    if (u_zero(b.Z)) { r = a; return; }
    fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, R, t;
    fe_sqr(Z1Z1, a.Z); fe_sqr(Z2Z2, b.Z);
    fe_mul(U1, a.X, Z2Z2); fe_mul(U2, b.X, Z1Z1);
    fe_mul(t, b.Z, Z2Z2); fe_mul(S1, a.Y, t);
    fe_mul(t, a.Z, Z1Z1); fe_mul(S2, b.Y, t);
    fe_sub(H, U2, U1); fe_sub(R, S2, S1);
    if (u_zero(H)) {
        if (u_zero(R)) { jac_dbl(r, a); return; }
        r = jac_inf(); return;
    }
    fe H2, H3, V, X3, Y3, Z3;
    fe_sqr(H2, H); fe_mul(H3, H, H2); fe_mul(V, U1, H2);
    fe_sqr(X3, R); fe_sub(X3, X3, H3); fe_sub(X3, X3, V); fe_sub(X3, X3, V);
    fe_sub(t, V, X3); fe_mul(Y3, R, t); fe_mul(t, S1, H3); fe_sub(Y3, Y3, t);
    fe_mul(Z3, a.Z, b.Z); fe_mul(Z3, Z3, H);
    r.X = X3; r.Y = Y3; r.Z = Z3;
}
__host__ __device__ inline Pt to_aff(const Jac& j) {
    Pt p; p.inf = false;
    if (u_zero(j.Z)) { p.inf = true; memset(&p.x, 0, sizeof p.x); memset(&p.y, 0, sizeof p.y); return p; }
    fe zi, z2, z3;
    fe_inv(zi, j.Z); fe_sqr(z2, zi); fe_mul(z3, z2, zi);
    fe_mul(p.x, j.X, z2); fe_mul(p.y, j.Y, z3);
    return p;
}
__host__ __device__ inline Pt pt_mul(const U256& k, const Pt& base) {
    Jac R = jac_inf(), B = jac_from(base);
    int nb = u_bits(k);
    for (int i = nb - 1; i >= 0; i--) {
        jac_dbl(R, R);
        if ((k.d[i / 64] >> (i % 64)) & 1) jac_add(R, R, B);
    }
    return to_aff(R);
}
__host__ __device__ __forceinline__ Pt mulG(const U256& k) { Pt g; g.x=GX; g.y=GY; g.inf=false; return pt_mul(k, g); }
__host__ __device__ __forceinline__ Pt pt_neg(const Pt& p) { Pt r = p; if (!p.inf) fe_neg(r.y, p.y); return r; }
__host__ __device__ __forceinline__ Pt pt_add(const Pt& a, const Pt& b) { Jac r; jac_add(r, jac_from(a), jac_from(b)); return to_aff(r); }
__host__ __device__ inline Pt mulG_i128(i128 k) {
    if (k >= 0) return mulG(u_from((u128)k));
    return pt_neg(mulG(u_from((u128)(-k))));
}
__host__ __device__ __forceinline__ bool pt_eq(const Pt& a, const Pt& b) {
    if (a.inf || b.inf) return a.inf == b.inf;
    return u_eq(a.x, b.x) && u_eq(a.y, b.y);
}
static U256 P_FIELD, EXP_SQRT;
static bool on_curve(const Pt& p) {
    fe y2, x3, rhs, seven; memset(&seven, 0, sizeof seven); seven.d[0] = 7;
    fe_sqr(y2, p.y); fe_sqr(x3, p.x); fe_mul(x3, x3, p.x); fe_add(rhs, x3, seven);
    return u_eq(y2, rhs);
}
static bool parse_pub(string s, Pt& P) {
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
    P.inf = false;
    if (s.size() == 66 && (s.compare(0, 2, "02") == 0 || s.compare(0, 2, "03") == 0)) {
        if (!hex2u(s.substr(2), P.x)) return false;
        if (u_cmp(P.x, P_FIELD) >= 0) return false;
        fe x3, rhs, seven, y, y2; memset(&seven, 0, sizeof seven); seven.d[0] = 7;
        fe_sqr(x3, P.x); fe_mul(x3, x3, P.x); fe_add(rhs, x3, seven);
        fe_pow(y, rhs, EXP_SQRT); fe_sqr(y2, y);
        if (!u_eq(y2, rhs)) return false;
        bool odd = y.d[0] & 1;
        if (odd != (s[1] == '3')) fe_neg(y, y);
        P.y = y; return true;
    }
    if (s.size() == 130 && s.compare(0, 2, "04") == 0) {
        if (!hex2u(s.substr(2, 64), P.x) || !hex2u(s.substr(66, 64), P.y)) return false;
        return on_curve(P);
    }
    return false;
}
static string pub_hex(const Pt& P) {
    string h = u2hex(P.x);
    return string((P.y.d[0] & 1) ? "03" : "02") + h;
}
static void init_curve() {
    hex2u("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", P_FIELD);
    hex2u("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", SECP_N);
    hex2u("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", GX);
    hex2u("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", GY);
    GEN.x = GX; GEN.y = GY; GEN.inf = false;
    U256 e = u_add(P_FIELD, u_from(1)); u_shr1(e); u_shr1(e); EXP_SQRT = e;  // (p+1)/4
}

// ============================== paramètres ===================================
static const int NJ = 256;     // taille de la table de sauts
static const int NE = 64;      // table de sauts "d'évasion" (cycles)
struct Jump { fe x, y; i128 d; };
static Jump g_J[NJ], g_E[NE];
#ifndef CUDA_STUB
__constant__ Jump d_J[NJ];
__constant__ Jump d_E[NE];
#endif

static U256 g_a, g_end, g_c;   // début, fin (incluse), centre
static u128 g_W = 0;           // largeur de l'intervalle
static int g_sym = 1;          // symétrie ±
static int g_dp = -1; static u64 g_dpmask = 0;
static int g_nthreads = 1, g_batch = 512;      // threads CPU
static double g_jf = -1, g_jg = -1; static int g_sl = 2;
static double g_coef = -1;
static double g_tfrac = 0.5;
static u64 g_seed = 0;
static u64 g_jhash = 0;

// options GPU
static int g_useGpu = -1;      // -1 auto
static int g_gBlocks = 256, g_gThreads = 256, g_gSteps = 48;

static vector<Pt> g_pub, g_Q;
static vector<string> g_pubstr;
static int g_nk = 0;
static atomic<int>* g_solved = nullptr;
static vector<i128> g_kval;
static atomic<int> g_nsolved{0};
static atomic<int> g_stop{0};
static atomic<u64> g_keyctr{0};
static mutex g_solmtx, g_relmtx, g_outmtx;
static string g_outfile = "kangaroo_trouve.txt";

// ---- RNG xoshiro-like, utilisable host ET device (chaque kangourou a son propre état) ----
struct Rng {
    u64 s[4];
    __host__ __device__ explicit Rng(u64 seed) {
        for (int i = 0; i < 4; i++) {
            seed += 0x9E3779B97F4A7C15ULL; u64 z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }
    __host__ __device__ static inline u64 rotl(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
    __host__ __device__ u64 next() {
        u64 r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    __host__ __device__ u128 next128() { u128 h = next(); return (h << 64) | next(); }
    __host__ __device__ u128 below(u128 n) { return next128() % n; }
};

static void build_jumps(double M) {
    Rng r(0xC0FFEE123456789ULL ^ (u64)g_W ^ (u64)(g_W >> 64));
    u128 span = (u128)(2.0L * (long double)M); if (span < 8) span = 8;
    g_jhash = (u64)span * 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < NJ + NE; i++) {
        Jump& j = (i < NJ) ? g_J[i] : g_E[i - NJ];
        j.d = (i128)(1 + r.below(span));
        Pt p = mulG_i128(j.d); j.x = p.x; j.y = p.y;
        g_jhash = (g_jhash ^ (u64)j.d) * 0x100000001B3ULL + (u64)(j.d >> 64);
    }
}

// ============================ table des DP (hote seul) ========================
static const u32 TAME = 0xFFFFFFFFu;
struct K128 { u64 a, b; bool operator==(const K128& o) const { return a == o.a && b == o.b; } };
struct KH { size_t operator()(const K128& k) const {
    u64 z = k.a * 0x9E3779B97F4A7C15ULL + k.b; z ^= z >> 29; z *= 0xBF58476D1CE4E5B9ULL; z ^= z >> 32; return (size_t)z; } };
struct Ent { i128 d; u32 id; int8_t s; };
struct Shard { mutex m; unordered_map<K128, Ent, KH> mp; };
static Shard g_sh[256];
static atomic<u64> g_ndp{0}, g_ntame{0};
struct DPRec { u64 x0, x1; i128 d; u32 id; int8_t s; };

struct Rel { int i, j, c; i128 k0; };      // k_i = c*k_j + k0
static vector<Rel> g_rel;

static void propagate();

static string hex_trim(const U256& a) {
    string h = u2hex(a); size_t p = h.find_first_not_of('0');
    return p == string::npos ? "0" : h.substr(p);
}
static void report_found(int idx, const U256& key) {
    string pub = g_pubstr[idx], priv = u2hex(key);
    printf("\n==============================================================\n"
           "  >>>  CLE TROUVEE  <<<                       (%d / %d)\n"
           "==============================================================\n"
           "  Cle publique : %s\n"
           "  Cle privee   : 0x%s\n"
           "  Enregistree  : %s\n"
           "%s"
           "==============================================================\n\n",
           g_nsolved.load(), g_nk, pub.c_str(), hex_trim(key).c_str(), g_outfile.c_str(),
           (u_cmp(key, g_a) < 0 || u_cmp(key, g_end) > 0) ? "  Note         : cle HORS de l'intervalle demande (verifiee par key*G)\n" : "");
    fflush(stdout);
    FILE* f = fopen(g_outfile.c_str(), "a");
    if (f) {
        time_t t = time(nullptr); char tb[64]; strftime(tb, sizeof tb, "%Y-%m-%d %H:%M:%S", localtime(&t));
        fprintf(f, "Date          : %s\nCle publique  : %s\nCle privee    : 0x%s\n\n", tb, pub.c_str(), priv.c_str());
        fclose(f);
    } else printf("ATTENTION : impossible d'ecrire dans %s\n", g_outfile.c_str());
}
static U256 key_from_k(i128 k) {
    if (k >= 0) {
        U256 kk = u_from((u128)k), r; u128 c = 0;
        for (int i = 0; i < 4; i++) { c += (u128)g_c.d[i] + kk.d[i]; r.d[i] = (u64)c; c >>= 64; }
        if (c || u_cmp(r, SECP_N) >= 0) r = u_sub(r, SECP_N);
        return r;
    }
    U256 kk = u_from((u128)(-k));
    if (u_cmp(g_c, kk) >= 0) return u_sub(g_c, kk);
    return u_sub(SECP_N, u_sub(kk, g_c));
}
static void try_key(int i, i128 k) {
    if (g_solved[i].load(memory_order_acquire)) return;
    U256 key = key_from_k(k);
    Pt P = mulG(key);
    if (!pt_eq(P, g_pub[i])) return;
    {
        lock_guard<mutex> lk(g_solmtx);
        if (g_solved[i].load(memory_order_acquire)) return;
        g_kval[i] = k;
        g_solved[i].store(1, memory_order_release);
        g_nsolved++;
        lock_guard<mutex> lo(g_outmtx);
        report_found(i, key);
        if (g_nsolved.load() >= g_nk) g_stop = 1;
    }
    propagate();
}
static void propagate() {
    bool ch = true;
    while (ch) {
        ch = false;
        vector<Rel> rel; { lock_guard<mutex> lk(g_relmtx); rel = g_rel; }
        for (auto& r : rel) {
            bool si = g_solved[r.i].load(memory_order_acquire), sj = g_solved[r.j].load(memory_order_acquire);
            if (si && !sj) { i128 kj = (i128)r.c * (g_kval[r.i] - r.k0); try_key(r.j, kj); if (g_solved[r.j].load()) ch = true; }
            else if (sj && !si) { i128 ki = (i128)r.c * g_kval[r.j] + r.k0; try_key(r.i, ki); if (g_solved[r.i].load()) ch = true; }
        }
    }
}
static void collide(const Ent& A, const Ent& B) {
    if (A.id == TAME && B.id == TAME) return;
    const Ent *t = nullptr, *w = nullptr;
    int nsig = g_sym ? 2 : 1;
    if (A.id == TAME) { t = &A; w = &B; } else if (B.id == TAME) { t = &B; w = &A; }
    if (t) {
        if (g_solved[w->id].load()) return;
        for (int q = 0; q < nsig; q++) {
            int sg = q ? -1 : 1;
            i128 k = (i128)w->s * ((i128)sg * t->d - w->d);
            try_key((int)w->id, k);
        }
        return;
    }
    for (int q = 0; q < nsig; q++) {
        int sg = q ? -1 : 1;
        if (A.id == B.id) {
            int coef = A.s - sg * B.s;
            if (coef == 0) continue;
            i128 rhs = (i128)sg * B.d - A.d;
            if (rhs & 1) continue;
            try_key((int)A.id, rhs / coef);
        } else {
            int c = A.s * sg * B.s;
            i128 k0 = (i128)A.s * ((i128)sg * B.d - A.d);
            bool si = g_solved[A.id].load(), sj = g_solved[B.id].load();
            if (si && sj) continue;
            { lock_guard<mutex> lk(g_relmtx); if (g_rel.size() < 2000000) g_rel.push_back({(int)A.id, (int)B.id, c, k0}); }
            if (si || sj) propagate();
        }
    }
}
static void add_dp(const DPRec& r) {
    K128 k{r.x0, r.x1};
    Shard& sh = g_sh[(r.x1 >> 8) & 255];
    Ent ne{r.d, r.id, r.s}, old;
    {
        lock_guard<mutex> lk(sh.m);
        auto it = sh.mp.find(k);
        if (it == sh.mp.end()) { sh.mp.emplace(k, ne); g_ndp++; if (r.id == TAME) g_ntame++; return; }
        old = it->second;
    }
    if (old.id == ne.id && old.d == ne.d) return;
    collide(old, ne);
}

// ======================== sauvegarde / chargement ============================
struct Hdr { char magic[8]; U256 a; u64 Wlo, Whi; u32 sym, dp; u64 jhash; u64 count; };
struct Rec { u64 x0, x1; i128 d; };

static bool save_tame(const string& fn) {
    vector<Rec> v; v.reserve(g_ntame.load());
    for (int i = 0; i < 256; i++) {
        lock_guard<mutex> lk(g_sh[i].m);
        for (auto& kv : g_sh[i].mp) if (kv.second.id == TAME) v.push_back({kv.first.a, kv.first.b, kv.second.d});
    }
    Hdr h; memset(&h, 0, sizeof h); memcpy(h.magic, "KANGDP01", 8);
    h.a = g_a; h.Wlo = (u64)g_W; h.Whi = (u64)(g_W >> 64); h.sym = g_sym; h.dp = g_dp; h.jhash = g_jhash; h.count = v.size();
    string tmp = fn + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { fprintf(stderr, "Erreur: impossible d'ecrire %s\n", tmp.c_str()); return false; }
    fwrite(&h, sizeof h, 1, f);
    if (!v.empty()) fwrite(v.data(), sizeof(Rec), v.size(), f);
    fclose(f);
    if (rename(tmp.c_str(), fn.c_str()) != 0) { fprintf(stderr, "Erreur: rename %s\n", fn.c_str()); return false; }
    return true;
}
static bool read_hdr(const string& fn, Hdr& h) {
    ifstream f(fn, ios::binary);
    if (!f) return false;
    f.read((char*)&h, sizeof h);
    return f.good() && memcmp(h.magic, "KANGDP01", 8) == 0;
}
static bool check_hdr(const Hdr& h, const string& fn) {
    if (!u_eq(h.a, g_a) || h.Wlo != (u64)g_W || h.Whi != (u64)(g_W >> 64)) {
        fprintf(stderr, "Fichier %s: intervalle different, refuse.\n", fn.c_str()); return false; }
    if ((int)h.sym != g_sym || h.jhash != g_jhash) {
        fprintf(stderr, "Fichier %s: parametres (symetrie / table de sauts) differents, refuse.\n", fn.c_str()); return false; }
    return true;
}
static bool load_tame(const string& fn, u64& n) {
    Hdr h;
    if (!read_hdr(fn, h)) { fprintf(stderr, "Fichier %s illisible.\n", fn.c_str()); return false; }
    if (!check_hdr(h, fn)) return false;
    if ((int)h.dp != g_dp) { fprintf(stderr, "Fichier %s: masque DP different (%u vs %d), refuse.\n", fn.c_str(), h.dp, g_dp); return false; }
    ifstream f(fn, ios::binary); f.seekg(sizeof(Hdr));
    vector<Rec> buf(1 << 16); u64 left = h.count; n = 0;
    while (left) {
        size_t m = (size_t)min<u64>(left, buf.size());
        f.read((char*)buf.data(), m * sizeof(Rec));
        if (!f) { fprintf(stderr, "Fichier %s tronque.\n", fn.c_str()); return false; }
        for (size_t i = 0; i < m; i++) add_dp({buf[i].x0, buf[i].x1, buf[i].d, TAME, 1});
        left -= m; n += m;
    }
    return true;
}

// ================================ worker CPU ==================================
struct alignas(64) Counter { atomic<u64> v; char pad[56]; };
static Counter* g_cnt = nullptr;      // sauts par "voie" (thread CPU ou GPU)
static Counter* g_tcnt = nullptr;     // sauts TAME par voie
static u64 g_rotLimit = 0;
static bool g_doTame = true, g_doWild = true;
static atomic<int> g_tameOff{0};

static int pick_key() {
    int n = g_nk; if (n == 0) return -1;
    for (int t = 0; t < n; t++) {
        int i = (int)(g_keyctr.fetch_add(1) % (u64)n);
        if (!g_solved[i].load()) return i;
    }
    return -1;
}

static const Jump* slow_add_cpu(fe& x, fe& y, const Jump* jp) {
    fe lam, t, x3, y3, inv;
    if (u_eq(x, jp->x)) {
        if (u_eq(y, jp->y)) {
            fe n, d; fe_sqr(n, x); fe_add(t, n, n); fe_add(n, t, n);
            fe_add(d, y, y); fe_inv(inv, d); fe_mul(lam, n, inv);
        } else {
            jp = &g_E[0];
            fe dx; fe_sub(dx, jp->x, x); fe_inv(inv, dx);
            fe_sub(t, jp->y, y); fe_mul(lam, t, inv);
        }
    } else {
        fe dx; fe_sub(dx, jp->x, x); fe_inv(inv, dx);
        fe_sub(t, jp->y, y); fe_mul(lam, t, inv);
    }
    fe l2; fe_sqr(l2, lam);
    fe_sub(x3, l2, x); fe_sub(x3, x3, jp->x);
    fe_sub(t, x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, y);
    x = x3; y = y3;
    return jp;
}

static void cpu_worker(int tid, int B) {
    Rng rng(g_seed * 0x9E3779B97F4A7C15ULL + (u64)tid * 7919ULL + 12345ULL);
    vector<fe> X(B), Y(B), DX(B), PRE(B);
    vector<i128> D(B);
    vector<u32> ID(B), SN(B), HP(B);
    vector<int8_t> S(B);
    vector<char> SLOW(B);
    vector<u64> H((size_t)B * 16), AGE(B);
    vector<const Jump*> JP(B);
    vector<DPRec> out; out.reserve(64);
    const u128 L = (g_sl >= 120) ? 1 : max<u128>(1, g_W >> g_sl);
    const u32 dpLimit = (u32)min<u64>(0xFFFFFFF0ULL, ((u64)64 << min(g_dp, 24)) + 4096);
    bool alive = true;

    auto roleTame = [&](int j) { return g_doTame && !g_tameOff.load(memory_order_relaxed) && (!g_doWild || (long)floor((j + 1) * g_tfrac) > (long)floor(j * g_tfrac)); };
    auto init = [&](int j) {
        bool tame = roleTame(j);
        for (;;) {
            i128 r = (i128)(rng.below(2 * L + 1)) - (i128)L;
            if (r == 0) r = 1;
            Pt P; int id = -1;
            if (tame) { P = mulG_i128(r); }
            else {
                id = pick_key();
                if (id < 0) { alive = false; return; }
                P = pt_add(g_Q[id], mulG_i128(r));
            }
            if (P.inf) continue;
            X[j] = P.x; Y[j] = P.y; D[j] = r; ID[j] = tame ? TAME : (u32)id; S[j] = 1;
            if (g_sym && (Y[j].d[0] & 1)) { fe_neg(Y[j], Y[j]); D[j] = -D[j]; S[j] = -1; }
            memset(&H[(size_t)j * 16], 0, 16 * sizeof(u64));
            HP[j] = 0; H[(size_t)j * 16 + (HP[j]++ & 15)] = X[j].d[0];
            SN[j] = 0; AGE[j] = 0;
            return;
        }
    };
    for (int j = 0; j < B && alive; j++) init(j);
    if (!alive) return;

    u64 iter = 0;
    while (!g_stop.load(memory_order_relaxed)) {
        for (int j = 0; j < B; j++) {
            const Jump* jp = nullptr;
            u64* h = &H[(size_t)j * 16];
            if (g_sym) {
                u32 pos = HP[j]; u64 xl = X[j].d[0];
                for (int Lc = 2; Lc <= 8; Lc += 2) {
                    if (h[(pos - 1 - Lc) & 15] == xl) {
                        u64 mn = xl;
                        for (int q = 1; q < Lc; q++) { u64 v = h[(pos - 1 - q) & 15]; if (v < mn) mn = v; }
                        jp = &g_E[(mn >> 7) % NE];
                        memset(h, 0, 16 * sizeof(u64)); HP[j] = 0; h[(HP[j]++) & 15] = xl;
                        break;
                    }
                }
            }
            if (!jp) jp = &g_J[X[j].d[1] & (NJ - 1)];
            JP[j] = jp;
            fe_sub(DX[j], jp->x, X[j]);
            if (u_zero(DX[j])) { SLOW[j] = 1; memset(&DX[j], 0, sizeof(fe)); DX[j].d[0] = 1; } else SLOW[j] = 0;
        }
        PRE[0] = DX[0];
        for (int j = 1; j < B; j++) fe_mul(PRE[j], PRE[j - 1], DX[j]);
        fe inv; fe_inv(inv, PRE[B - 1]);
        for (int j = B - 1; j > 0; j--) {
            fe t; fe_mul(t, inv, PRE[j - 1]);
            fe_mul(inv, inv, DX[j]);
            DX[j] = t;
        }
        DX[0] = inv;
        u64 tameStep = 0;
        for (int j = 0; j < B; j++) {
            const Jump* jp = JP[j];
            if (SLOW[j]) jp = slow_add_cpu(X[j], Y[j], jp);
            else {
                fe t, lam, l2, x3, y3;
                fe_sub(t, jp->y, Y[j]); fe_mul(lam, t, DX[j]);
                fe_sqr(l2, lam);
                fe_sub(x3, l2, X[j]); fe_sub(x3, x3, jp->x);
                fe_sub(t, X[j], x3); fe_mul(y3, lam, t); fe_sub(y3, y3, Y[j]);
                X[j] = x3; Y[j] = y3;
            }
            D[j] += jp->d;
            if (g_sym && (Y[j].d[0] & 1)) { fe_neg(Y[j], Y[j]); D[j] = -D[j]; S[j] = -S[j]; }
            u64* h = &H[(size_t)j * 16];
            h[(HP[j]++) & 15] = X[j].d[0];
            SN[j]++; AGE[j]++;
            if (ID[j] == TAME) tameStep++;
            if ((X[j].d[0] & g_dpmask) == 0) {
                out.push_back({X[j].d[0], X[j].d[1], D[j], ID[j], S[j]});
                SN[j] = 0;
            } else if (SN[j] > dpLimit) {
                init(j); if (!alive) break;
            }
        }
        if (!alive) break;
        g_cnt[tid].v.fetch_add((u64)B, memory_order_relaxed);
        if (tameStep) g_tcnt[tid].v.fetch_add(tameStep, memory_order_relaxed);
        if (!out.empty()) { for (auto& r : out) add_dp(r); out.clear(); }
        iter++;
        if ((iter & 63) == 0) {
            for (int j = 0; j < B; j++) {
                if (ID[j] == TAME) { if (g_tameOff.load(memory_order_relaxed)) { init(j); if (!alive) break; } }
                else if (g_solved[ID[j]].load() || (g_rotLimit && AGE[j] > g_rotLimit)) { init(j); if (!alive) break; }
            }
            if (!alive) break;
        }
    }
}

// ================================ GPU : noyau kangourou ========================
// Un thread CUDA = un kangourou. Simplification volontaire par rapport au CPU :
// pas d'inversion par lots (Montgomery batch trick), chaque thread fait sa
// propre fe_inv. Moins optimal que le CPU à nombre de kangourous égal, mais
// massivement parallèle (dizaines/centaines de milliers de kangourous
// simultanés). Piste d'optimisation future : inversion par lots au niveau
// du block (scan parallèle "produit sauf soi-même" en mémoire partagée).
#ifndef CUDA_STUB

static const u32 GPU_INACTIVE = 0xFFFFFFFEu;   // sentinelle : plus de cle a traiter

struct DPOut { u64 x0, x1; i128 d; u32 id; int8_t s; };

__device__ __forceinline__ void gpu_reinit(
    fe& x, fe& y, i128& d, u32& id, int8_t& s, u64 (&h)[16], u32& hp, u32& sn, u64& age,
    Rng& rng, long gidx, const Pt* Qdev, int nk, const u8* solvedDev, unsigned long long* keyCtr,
    u128 L, double tfrac, int doTame, int doWild, int tameOff, int sym)
{
    bool tame = doTame && !tameOff && (!doWild || (long)floor(((double)gidx + 1) * tfrac) > (long)floor((double)gidx * tfrac));
    for (;;) {
        u128 rr = rng.below(2 * L + 1);
        i128 r = (i128)rr - (i128)L; if (r == 0) r = 1;
        Pt P; int kid = -1;
        if (tame) P = mulG_i128(r);
        else {
            kid = -1;
            for (int t = 0; t < nk; t++) {
                unsigned long long c = atomicAdd(keyCtr, 1ULL);
                int cand = (int)(c % (unsigned long long)nk);
                if (!solvedDev[cand]) { kid = cand; break; }
            }
            if (kid < 0) { id = GPU_INACTIVE; return; }
            P = pt_add(Qdev[kid], mulG_i128(r));
        }
        if (P.inf) continue;
        x = P.x; y = P.y; d = r; id = tame ? TAME : (u32)kid; s = 1;
        if (sym && (y.d[0] & 1)) { fe_neg(y, y); d = -d; s = -1; }
        for (int k = 0; k < 16; k++) h[k] = 0;
        hp = 0; h[hp++ & 15] = x.d[0];
        sn = 0; age = 0;
        return;
    }
}

__global__ void k_init(fe* X, fe* Y, i128* D, u32* ID, int8_t* S, u64* H, u32* HP, u32* SN, u64* AGE,
                        u64* RS, int N, const Pt* Qdev, int nk, const u8* solvedDev, unsigned long long* keyCtr,
                        u128 L, double tfrac, int doTame, int doWild, u64 seedBase, int sym)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    Rng rng(seedBase ^ ((u64)idx * 0x9E3779B97F4A7C15ULL) ^ 0xD1B54A32D192ED03ULL);
    fe x, y; i128 d; u32 id; int8_t s; u64 h[16]; u32 hp, sn; u64 age;
    gpu_reinit(x, y, d, id, s, h, hp, sn, age, rng, idx, Qdev, nk, solvedDev, keyCtr, L, tfrac, doTame, doWild, 0, sym);
    X[idx] = x; Y[idx] = y; D[idx] = d; ID[idx] = id; S[idx] = s;
    for (int k = 0; k < 16; k++) H[(size_t)idx * 16 + k] = h[k];
    HP[idx] = hp; SN[idx] = sn; AGE[idx] = age;
    RS[idx * 4] = rng.s[0]; RS[idx * 4 + 1] = rng.s[1]; RS[idx * 4 + 2] = rng.s[2]; RS[idx * 4 + 3] = rng.s[3];
}

__global__ void k_kangaroo(fe* X, fe* Y, i128* D, u32* ID, int8_t* S, u64* H, u32* HP, u32* SN, u64* AGE, u64* RS,
                            int N, int steps, u64 dpmask, u32 dpLimit, int sym,
                            DPOut* dpBuf, unsigned long long* dpCount, unsigned long long dpCap,
                            unsigned long long* stepCtr, unsigned long long* tameCtr,
                            const Pt* Qdev, int nk, const u8* solvedDev, unsigned long long* keyCtr,
                            u128 L, double tfrac, int doTame, int doWild, int tameOff, u64 rotLimit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    fe x = X[idx], y = Y[idx]; i128 d = D[idx]; u32 id = ID[idx]; int8_t s = S[idx];
    u64 h[16]; for (int k = 0; k < 16; k++) h[k] = H[(size_t)idx * 16 + k];
    u32 hp = HP[idx], sn = SN[idx]; u64 age = AGE[idx];
    Rng rng(0); rng.s[0] = RS[idx * 4]; rng.s[1] = RS[idx * 4 + 1]; rng.s[2] = RS[idx * 4 + 2]; rng.s[3] = RS[idx * 4 + 3];

    unsigned long long executed = 0, tameExec = 0;
    for (int it = 0; it < steps; it++) {
        if (id == GPU_INACTIVE) break;
        if (id == TAME) {
            if (tameOff) { gpu_reinit(x, y, d, id, s, h, hp, sn, age, rng, idx, Qdev, nk, solvedDev, keyCtr, L, tfrac, doTame, doWild, tameOff, sym); if (id == GPU_INACTIVE) break; }
        } else if ((int)id < nk) {
            if (solvedDev[id] || (rotLimit && age > rotLimit)) { gpu_reinit(x, y, d, id, s, h, hp, sn, age, rng, idx, Qdev, nk, solvedDev, keyCtr, L, tfrac, doTame, doWild, tameOff, sym); if (id == GPU_INACTIVE) break; }
        }

        const Jump* jp = nullptr;
        if (sym) {
            u64 xl = x.d[0];
            for (int Lc = 2; Lc <= 8; Lc += 2) {
                if (h[(hp - 1 - Lc) & 15] == xl) {
                    u64 mn = xl;
                    for (int q = 1; q < Lc; q++) { u64 v = h[(hp - 1 - q) & 15]; if (v < mn) mn = v; }
                    jp = &d_E[(mn >> 7) % NE];
                    for (int k = 0; k < 16; k++) h[k] = 0; hp = 0; h[(hp++) & 15] = xl;
                    break;
                }
            }
        }
        if (!jp) jp = &d_J[x.d[1] & (NJ - 1)];

        fe dx; fe_sub(dx, jp->x, x);
        fe x3, y3;
        if (u_zero(dx)) {
            fe lam, t, inv;
            if (u_eq(x, jp->x) && u_eq(y, jp->y)) {
                fe n, dd; fe_sqr(n, x); fe_add(t, n, n); fe_add(n, t, n);
                fe_add(dd, y, y); fe_inv(inv, dd); fe_mul(lam, n, inv);
            } else {
                const Jump* jp2 = &d_E[0];
                fe dx2; fe_sub(dx2, jp2->x, x); fe_inv(inv, dx2);
                fe_sub(t, jp2->y, y); fe_mul(lam, t, inv);
                jp = jp2;
            }
            fe l2; fe_sqr(l2, lam);
            fe_sub(x3, l2, x); fe_sub(x3, x3, jp->x);
            fe_sub(t, x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, y);
        } else {
            fe inv; fe_inv(inv, dx);
            fe t, lam, l2;
            fe_sub(t, jp->y, y); fe_mul(lam, t, inv);
            fe_sqr(l2, lam);
            fe_sub(x3, l2, x); fe_sub(x3, x3, jp->x);
            fe_sub(t, x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, y);
        }
        x = x3; y = y3;
        d += jp->d;
        if (sym && (y.d[0] & 1)) { fe_neg(y, y); d = -d; s = -s; }
        h[(hp++) & 15] = x.d[0];
        sn++; age++; executed++;
        if (id == TAME) tameExec++;
        if ((x.d[0] & dpmask) == 0) {
            unsigned long long pos = atomicAdd(dpCount, 1ULL);
            if (pos < dpCap) { dpBuf[pos].x0 = x.d[0]; dpBuf[pos].x1 = x.d[1]; dpBuf[pos].d = d; dpBuf[pos].id = id; dpBuf[pos].s = s; }
            sn = 0;
        } else if (sn > dpLimit) {
            gpu_reinit(x, y, d, id, s, h, hp, sn, age, rng, idx, Qdev, nk, solvedDev, keyCtr, L, tfrac, doTame, doWild, tameOff, sym);
            if (id == GPU_INACTIVE) break;
        }
    }
    X[idx] = x; Y[idx] = y; D[idx] = d; ID[idx] = id; S[idx] = s;
    for (int k = 0; k < 16; k++) H[(size_t)idx * 16 + k] = h[k];
    HP[idx] = hp; SN[idx] = sn; AGE[idx] = age;
    RS[idx * 4] = rng.s[0]; RS[idx * 4 + 1] = rng.s[1]; RS[idx * 4 + 2] = rng.s[2]; RS[idx * 4 + 3] = rng.s[3];
    if (executed) atomicAdd(stepCtr, executed);
    if (tameExec) atomicAdd(tameCtr, tameExec);
}

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "Erreur CUDA %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static void gpu_worker(int gpuId, int cntIdx) {
    CUDA_CHECK(cudaSetDevice(gpuId));
    CUDA_CHECK(cudaMemcpyToSymbol(d_J, g_J, sizeof(g_J)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_E, g_E, sizeof(g_E)));

    const int N = g_gBlocks * g_gThreads;
    fe *X, *Y; i128 *D; u32 *ID; int8_t *S; u64 *H, *RS, *AGE; u32 *HP, *SN;
    DPOut* dpBuf; unsigned long long *dpCount, *stepCtr, *tameCtr, *keyCtr;
    Pt* Qdev; u8* solvedDev;
    const unsigned long long dpCap = (unsigned long long)N;

    CUDA_CHECK(cudaMalloc(&X, N * sizeof(fe)));
    CUDA_CHECK(cudaMalloc(&Y, N * sizeof(fe)));
    CUDA_CHECK(cudaMalloc(&D, N * sizeof(i128)));
    CUDA_CHECK(cudaMalloc(&ID, N * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&S, N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&H, (size_t)N * 16 * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&RS, (size_t)N * 4 * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&AGE, N * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&HP, N * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&SN, N * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&dpBuf, dpCap * sizeof(DPOut)));
    CUDA_CHECK(cudaMalloc(&dpCount, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&stepCtr, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&tameCtr, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&keyCtr, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(keyCtr, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&Qdev, max(1, g_nk) * sizeof(Pt)));
    if (g_nk) CUDA_CHECK(cudaMemcpy(Qdev, g_Q.data(), g_nk * sizeof(Pt), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&solvedDev, max(1, g_nk)));

    const u128 L = (g_sl >= 120) ? 1 : max<u128>(1, g_W >> g_sl);
    const u32 dpLimit = (u32)min<u64>(0xFFFFFFF0ULL, ((u64)64 << min(g_dp, 24)) + 4096);
    vector<u8> solvedHost(max(1, g_nk), 0);

    dim3 blk(g_gThreads), grd(g_gBlocks);
    u64 seedBase = g_seed * 0x9E3779B97F4A7C15ULL ^ ((u64)(gpuId + 1) * 0xBF58476D1CE4E5B9ULL);
    k_init<<<grd, blk>>>(X, Y, D, ID, S, H, HP, SN, AGE, RS, N, Qdev, g_nk, solvedDev, keyCtr,
                          L, g_tfrac, g_doTame ? 1 : 0, g_doWild ? 1 : 0, seedBase, g_sym);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<DPOut> hostBuf(dpCap);
    while (!g_stop.load(memory_order_relaxed)) {
        for (int i = 0; i < g_nk; i++) solvedHost[i] = (u8)g_solved[i].load(memory_order_relaxed);
        if (g_nk) CUDA_CHECK(cudaMemcpy(solvedDev, solvedHost.data(), g_nk, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dpCount, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(stepCtr, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(tameCtr, 0, sizeof(unsigned long long)));

        k_kangaroo<<<grd, blk>>>(X, Y, D, ID, S, H, HP, SN, AGE, RS, N, g_gSteps, g_dpmask, dpLimit, g_sym,
                                  dpBuf, dpCount, dpCap, stepCtr, tameCtr, Qdev, g_nk, solvedDev, keyCtr,
                                  L, g_tfrac, g_doTame ? 1 : 0, g_doWild ? 1 : 0, g_tameOff.load() ? 1 : 0, g_rotLimit);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long cnt = 0, steps_ = 0, tame_ = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, dpCount, sizeof(cnt), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&steps_, stepCtr, sizeof(steps_), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&tame_, tameCtr, sizeof(tame_), cudaMemcpyDeviceToHost));
        g_cnt[cntIdx].v.fetch_add(steps_, memory_order_relaxed);
        if (tame_) g_tcnt[cntIdx].v.fetch_add(tame_, memory_order_relaxed);
        unsigned long long got = min(cnt, dpCap);
        if (got) {
            CUDA_CHECK(cudaMemcpy(hostBuf.data(), dpBuf, got * sizeof(DPOut), cudaMemcpyDeviceToHost));
            for (unsigned long long i = 0; i < got; i++) {
                DPRec r{hostBuf[i].x0, hostBuf[i].x1, hostBuf[i].d, hostBuf[i].id, hostBuf[i].s};
                if (r.id == TAME || (int)r.id < g_nk) add_dp(r);   // ignore toute entree aberrante
            }
        }
    }
    cudaFree(X); cudaFree(Y); cudaFree(D); cudaFree(ID); cudaFree(S); cudaFree(H); cudaFree(RS);
    cudaFree(AGE); cudaFree(HP); cudaFree(SN); cudaFree(dpBuf); cudaFree(dpCount); cudaFree(stepCtr);
    cudaFree(tameCtr); cudaFree(keyCtr); cudaFree(Qdev); cudaFree(solvedDev);
}
static int gpu_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}
#else
static void gpu_worker(int, int) {}
static int gpu_device_count() { return 0; }
#endif

// ================================= main ======================================
static void on_sigint(int) { g_stop = 1; }
static string fmt_time(double s) {
    long t = (long)s; char b[64]; snprintf(b, sizeof b, "%02ld:%02ld:%02ld", t / 3600, (t / 60) % 60, t % 60); return b;
}
static string fmt_bytes(double b) {
    const char* u[] = {"o", "Ko", "Mo", "Go", "To"}; int i = 0;
    while (b >= 1024.0 && i < 4) { b /= 1024.0; i++; }
    char s[48]; snprintf(s, sizeof s, "%.1f %s", b, u[i]); return s;
}
static double dp_ram_bytes() { return (double)sizeof(pair<const K128, Ent>) + 40.0; }

static void usage() {
    puts("Kangaroo secp256k1 (GPU + CPU)\n"
         "Usage :\n"
         "  -pub <hex>            cle publique (02/03.. ou 04..) [alias -cle_publique]\n"
         "  -fichier <fichier>    fichier de cles publiques (une par ligne)\n"
         "  -r <debut:fin>        intervalle en hexadecimal (ou <debut> avec -b)\n"
         "  -b <bits>             largeur en bits\n"
         "  -bb <bits>            intervalle complet [1, 2^bb - 1]\n"
         "  -s <fichier>          fichier d'enregistrement de la table tame\n"
         "  -ss <f1[,f2..]>       fichier(s) tame a charger\n"
         "  -t <sec>              intervalle d'affichage (defaut 30)\n"
         "Options CPU :\n"
         "  -c <n>      threads CPU (defaut : coeurs - 1 si GPU dispo, sinon tous les coeurs)\n"
         "  -k <n>      kangourous par thread CPU (defaut 512)\n"
         "Options GPU :\n"
         "  -gpu 0|1    force desactiver/activer le GPU (defaut : auto)\n"
         "  -gb <n>     blocks CUDA par GPU (defaut 256)\n"
         "  -gt <n>     threads par block CUDA (defaut 256) -> kangourous GPU = gb*gt par GPU\n"
         "  -gs <n>     pas par appel de noyau avant releve des DP (defaut 48)\n"
         "Autres options : -end, -dp, -nosym, -tame, -mix/-both, -tw, -sv, -duree, -o, -genpub, -test\n"
         "  (voir le code pour le detail, identique a la version CPU d'origine)");
}

static int selftest() {
    int bad = 0;
    fe a; hex2u("123456789abcdef0fedcba98765432100f1e2d3c4b5a69788796a5b4c3d2e1f0", a);
    fe i, p; fe_inv(i, a); fe_mul(p, a, i);
    fe one; memset(&one, 0, sizeof one); one.d[0] = 1;
    if (!u_eq(p, one)) { puts("FAIL inv"); bad++; }
    U256 e = u_sub(P_FIELD, u_from(2)); fe i2; fe_pow(i2, a, e);
    if (!u_eq(i, i2)) { puts("FAIL inv vs pow"); bad++; }
    Pt g2 = mulG(u_from(2));
    U256 x2; hex2u("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5", x2);
    if (!u_eq(g2.x, x2)) { puts("FAIL 2G"); bad++; }
    Pt g3 = pt_add(g2, GEN), g3b = mulG(u_from(3));
    if (!pt_eq(g3, g3b)) { puts("FAIL 3G"); bad++; }
    Pt n1 = mulG(u_sub(SECP_N, u_from(1)));
    if (!pt_eq(n1, pt_neg(GEN))) { puts("FAIL (n-1)G"); bad++; }
    Pt pp; if (!parse_pub(pub_hex(g3b), pp) || !pt_eq(pp, g3b)) { puts("FAIL parse"); bad++; }
    printf(bad ? "Auto-tests : %d echec(s)\n" : "Auto-tests OK\n", bad);
    return bad;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    init_curve();
    string pubArg, fileArg, rArg, saveFile, loadArg;
    bool mixMode = false, resumeSave = false;
    int bits = 0, bb = 0, tsec = 30, duree = 0; double endCoef = 0; bool nosym = false, forceTame = false;
    int cores = (int)thread::hardware_concurrency(); if (cores < 1) cores = 1;
    bool coresSetByUser = false;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto nxt = [&]() -> string { if (i + 1 >= argc) { fprintf(stderr, "Argument manquant apres %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "-pub" || a == "-cle_publique") pubArg = nxt();
        else if (a == "-fichier") fileArg = nxt();
        else if (a == "-r") rArg = nxt();
        else if (a == "-b") bits = atoi(nxt().c_str());
        else if (a == "-bb") bb = atoi(nxt().c_str());
        else if (a == "-end") { endCoef = atof(nxt().c_str()); if (!(endCoef > 0)) { fprintf(stderr, "-end doit etre > 0\n"); return 1; } }
        else if (a == "-s") saveFile = nxt();
        else if (a == "-ss") loadArg = nxt();
        else if (a == "-sv") { saveFile = nxt(); resumeSave = true; }
        else if (a == "-t") tsec = max(1, atoi(nxt().c_str()));
        else if (a == "-c") { cores = max(0, atoi(nxt().c_str())); coresSetByUser = true; }
        else if (a == "-k") g_batch = max(2, atoi(nxt().c_str()));
        else if (a == "-dp") g_dp = atoi(nxt().c_str());
        else if (a == "-nosym") nosym = true;
        else if (a == "-tame") forceTame = true;
        else if (a == "-mix" || a == "-both") mixMode = true;
        else if (a == "-duree") duree = atoi(nxt().c_str());
        else if (a == "-o") g_outfile = nxt();
        else if (a == "-seed") g_seed = strtoull(nxt().c_str(), nullptr, 10);
        else if (a == "-jf") g_jf = atof(nxt().c_str());
        else if (a == "-jg") g_jg = atof(nxt().c_str());
        else if (a == "-sl") g_sl = atoi(nxt().c_str());
        else if (a == "-coef") g_coef = atof(nxt().c_str());
        else if (a == "-tw") g_tfrac = atof(nxt().c_str());
        else if (a == "-gpu") g_useGpu = atoi(nxt().c_str());
        else if (a == "-gb") g_gBlocks = max(1, atoi(nxt().c_str()));
        else if (a == "-gt") g_gThreads = max(32, atoi(nxt().c_str()));
        else if (a == "-gs") g_gSteps = max(1, atoi(nxt().c_str()));
        else if (a == "-test") return selftest();
        else if (a == "-genpub") {
            U256 k; if (!hex2u(nxt(), k)) { fprintf(stderr, "hex invalide\n"); return 1; }
            printf("%s\n", pub_hex(mulG(k)).c_str()); return 0;
        }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "Option inconnue: %s\n", a.c_str()); usage(); return 1; }
    }
    if (!g_seed) g_seed = (u64)chrono::steady_clock::now().time_since_epoch().count() ^ 0xA5A5A5A5ULL;
    g_sym = nosym ? 0 : 1;

    int nGpu = gpu_device_count();
    if (g_useGpu == 0) nGpu = 0;
    if (!coresSetByUser) cores = (nGpu > 0) ? max(0, cores - 1) : cores;
    g_nthreads = cores;
    if (g_coef < 0) g_coef = g_sym ? 1.5 : 2.8;

    {
        U256 a, e; memset(&a, 0, sizeof a);
        size_t colon = rArg.find(':');
        if (colon != string::npos) {
            if (!hex2u(rArg.substr(0, colon), a) || !hex2u(rArg.substr(colon + 1), e)) { fprintf(stderr, "Intervalle -r invalide\n"); return 1; }
        } else if (!rArg.empty()) {
            int wb = bits >= 8 ? bits : bb;
            if (wb < 8) { fprintf(stderr, "-r <debut> demande aussi -b <bits>\n"); return 1; }
            if (wb > 120) { fprintf(stderr, "Largeur trop grande (max 2^120).\n"); return 1; }
            if (!hex2u(rArg, a)) { fprintf(stderr, "Debut -r invalide\n"); return 1; }
            U256 w = u_from(((u128)1 << wb) - 1); e = u_add(a, w);
        } else if (bb > 0) {
            if (bb < 9 || bb > 120) { fprintf(stderr, "-bb doit etre compris entre 9 et 120\n"); return 1; }
            a = u_from(1); e = u_from(((u128)1 << bb) - 1);
        } else if (bits >= 8) {
            a = u_from((u128)1 << (bits - 1)); e = u_from(((u128)1 << bits) - 1);
        } else { usage(); return 1; }
        if (u_cmp(e, a) <= 0 || u_cmp(e, SECP_N) >= 0) { fprintf(stderr, "Intervalle invalide (fin <= debut ou >= n)\n"); return 1; }
        U256 w = u_add(u_sub(e, a), u_from(1));
        if (w.d[2] || w.d[3] || w.d[1] >> 56) { fprintf(stderr, "Intervalle trop large (max ~2^120).\n"); return 1; }
        g_a = a; g_end = e; g_W = ((u128)w.d[1] << 64) | w.d[0];
        if (g_W < 256) { fprintf(stderr, "Intervalle trop petit (min 2^8).\n"); return 1; }
        g_c = u_add(a, u_from(g_W / 2));
    }
    double sqW = sqrt((double)g_W);

    vector<string> lines;
    if (!pubArg.empty()) lines.push_back(pubArg);
    if (!fileArg.empty()) {
        ifstream f(fileArg); if (!f) { fprintf(stderr, "Impossible d'ouvrir %s\n", fileArg.c_str()); return 1; }
        string l; while (getline(f, l)) {
            size_t h = l.find('#'); if (h != string::npos) l = l.substr(0, h);
            while (!l.empty() && isspace((unsigned char)l.back())) l.pop_back();
            size_t s0 = 0; while (s0 < l.size() && isspace((unsigned char)l[s0])) s0++;
            l = l.substr(s0); if (!l.empty()) lines.push_back(l);
        }
    }
    for (auto& l : lines) {
        Pt P;
        if (!parse_pub(l, P)) { fprintf(stderr, "Cle publique invalide ignoree: %s\n", l.c_str()); continue; }
        bool dup = false; for (auto& q : g_pub) if (pt_eq(q, P)) dup = true;
        if (dup) continue;
        g_pub.push_back(P); g_pubstr.push_back(l);
    }
    g_nk = (int)g_pub.size();
    if (g_nk == 0 && saveFile.empty()) { fprintf(stderr, "Aucune cle valide et pas de -s : rien a faire.\n"); usage(); return 1; }
    g_solved = new atomic<int>[max(1, g_nk)]; for (int i = 0; i < max(1, g_nk); i++) g_solved[i] = 0;
    g_kval.assign(max(1, g_nk), 0);

    double jf = g_jf, jg = g_jg;
    if (jf < 0) jf = g_sym ? 1.0 : 0.7;
    if (jg < 0) jg = g_sym ? 0.75 : 0.5;
    double M = max(4.0, jf * pow((double)g_W, jg));
    build_jumps(M);

    {
        Pt cG = mulG(g_c), ncG = pt_neg(cG);
        for (int i = 0; i < g_nk; i++) {
            Pt Q = pt_add(g_pub[i], ncG);
            if (Q.inf) {
                g_solved[i] = 1; g_kval[i] = 0; g_nsolved++;
                report_found(i, g_c);
                Q = g_pub[i];
            }
            g_Q.push_back(Q);
        }
        if (g_nk > 0 && g_nsolved >= g_nk) return 0;
    }

    vector<string> loadFiles;
    { stringstream ss(loadArg); string t; while (getline(ss, t, ',')) if (!t.empty()) loadFiles.push_back(t); }
    if (resumeSave) {
        ifstream ex(saveFile, ios::binary);
        if (ex) {
            Hdr h0;
            if (!read_hdr(saveFile, h0)) { fprintf(stderr, "-sv: %s existe mais invalide : refuse.\n", saveFile.c_str()); return 1; }
            loadFiles.push_back(saveFile);
        }
    }
    bool dpFromFile = false;
    if (!loadFiles.empty() && g_dp < 0) {
        Hdr h; if (read_hdr(loadFiles[0], h)) { g_dp = (int)h.dp; dpFromFile = true; }
    }
    g_doWild = g_nk > 0;
    g_doTame = (g_nk == 0) || loadFiles.empty() || forceTame || resumeSave || mixMode;
    int KtotCpu = g_nthreads * g_batch;
    int KtotGpu = nGpu * g_gBlocks * g_gThreads;
    int Ktot = max(1, KtotCpu + KtotGpu);
    double Nexp = g_coef * sqW * sqrt((double)max(1, g_nk));
    const bool tameGoal = (endCoef > 0 && g_doTame && !g_doWild);
    const bool mixGoal = (endCoef > 0 && mixMode && g_doTame && g_doWild);
    double endTarget = 0, baseSteps = 0;
    if (tameGoal) { endTarget = endCoef * sqW; Nexp = endTarget; }
    if (mixGoal) { endTarget = endCoef * sqW; }
    const bool dpAuto = (g_dp < 0);
    if (g_dp < 0) {
        double lN = log2(Nexp);
        int dpTarget = (int)floor(lN) - 22;
        int dpOver = (int)floor(log2(Nexp / (50.0 * Ktot)));
        int dpRam = (int)ceil(lN) - 23;
        g_dp = max(0, min(dpTarget, dpOver));
        g_dp = max(g_dp, dpRam);
        if (g_dp > 60) g_dp = 60;
    }
    if (g_dp < 0) g_dp = 0;
    g_dpmask = (g_dp >= 64) ? ~0ULL : ((1ULL << g_dp) - 1);
    {
        double dpGuess = (Nexp + sqW) / (double)(1ULL << min(g_dp, 40));
        size_t perShard = (size_t)max(16.0, 1.3 * dpGuess / 256.0);
        for (int i = 0; i < 256; i++) g_sh[i].mp.reserve(perShard);
    }
    for (auto& fn : loadFiles) {
        u64 n = 0; if (!load_tame(fn, n)) return 1;
        printf("Charge %s : %llu DP tame\n", fn.c_str(), (unsigned long long)n);
    }
    if ((tameGoal || mixGoal) && !loadFiles.empty())
        baseSteps = min(endTarget, (double)g_ntame.load() * ldexp(1.0, min(g_dp, 62)));
    if (resumeSave && !loadFiles.empty()) {
        string bak = saveFile + ".bak";
        ifstream in(saveFile, ios::binary); ofstream out(bak, ios::binary | ios::trunc);
        out << in.rdbuf();
        printf("Reprise -sv : copie de securite %s\n", bak.c_str());
    }
    int nwild = g_doWild ? (g_doTame ? (int)(Ktot * (1.0 - g_tfrac)) : Ktot) : 0;
    g_rotLimit = (g_nk > nwild && nwild > 0) ? ((u64)256 << min(g_dp, 24)) + 8192 : 0;

    printf("\n==============================================================\n"
           "  KANGAROO secp256k1  -  GPU + CPU\n"
           "==============================================================\n");
    printf("  Intervalle     : 0x%s\n               -> 0x%s\n", hex_trim(g_a).c_str(), hex_trim(g_end).c_str());
    printf("  Largeur        : 2^%.1f\n", log2((double)g_W));
    printf("  Cles           : %d\n", g_nk);
    printf("  GPU detectes   : %d%s\n", nGpu, nGpu ? "" : "  (aucun -> calcul CPU uniquement)");
    if (nGpu) printf("  Kangourous GPU : %d par GPU (%d blocks x %d threads), %d au total\n", g_gBlocks * g_gThreads, g_gBlocks, g_gThreads, KtotGpu);
    printf("  Threads CPU    : %d x %d kangourous%s\n", g_nthreads, g_batch, g_nthreads == 0 ? "  (CPU desactive, -c 0)" : "");
    printf("  Symetrie       : %s\n", g_sym ? "oui" : "non");
    const double pDP = ldexp(1.0, -min(g_dp, 1000));
    const double dpEst = Nexp * pDP;
    const double tameShare = !g_doTame ? 0.0 : (g_doWild ? g_tfrac : 1.0);
    const double dpTameEst = Nexp * tameShare * pDP;
    const double ovSteps = (double)Ktot * ldexp(1.0, min(g_dp, 1000));
    const double ovPct = 100.0 * ovSteps / Nexp;
    printf("  Points DP      : %d bits (%s) -> 1 point sur %.0f\n", g_dp, dpAuto ? "auto" : (dpFromFile ? "repris du fichier" : "manuel"), ldexp(1.0, min(g_dp, 62)));
    printf("  DP attendus    : ~%.3g au total (tame ~%.3g, wild ~%.3g) apres ~%.3g sauts\n", dpEst, dpTameEst, dpEst - dpTameEst, Nexp);
    printf("  Memoire DP est.: ~%s  (%.0f octets/DP en RAM)\n", fmt_bytes(dpEst * dp_ram_bytes()).c_str(), dp_ram_bytes());
    if (!saveFile.empty() && g_doTame)
        printf("  Fichier tame   : ~%s estimes\n", fmt_bytes(sizeof(Hdr) + dpTameEst * sizeof(Rec)).c_str());
    printf("  Surcout DP     : ~%.3g sauts (%.2f %% de la moyenne)\n", ovSteps, ovPct);
    if (ovPct > 10.0) printf("  /!\\ Surcout DP eleve (%.1f %%) : diminuez -dp, -gs ou -k/-c/-gb/-gt\n", ovPct);
    if (dpEst > 3.4e7) printf("  /!\\ Beaucoup de DP (~%.3g) : augmentez -dp\n", dpEst);
    if (dpEst < 1000) printf("  /!\\ Tres peu de DP attendus (~%.0f) : diminuez -dp\n", dpEst);
    printf("  Saut moyen     : 2^%.1f\n", log2(M));
    printf("  Mode           : %s%s%s\n", g_doTame ? "tame" : "", (g_doTame && g_doWild) ? " + " : "", g_doWild ? "wild" : "");
    if (tameGoal) printf("  Cible tame     : %.4g x sqrt(W) = ~%.3g sauts\n", endCoef, endTarget);
    else if (g_nk > 0) printf("  Sauts attendus : ~%.3g (moyenne)\n", Nexp);
    if (mixMode && !g_doWild) printf("  -mix ignore    : il faut au moins une cle publique\n");
    if (g_nk > 0) printf("  Resultats dans : %s\n", g_outfile.c_str());
    if (!saveFile.empty()) printf("  Sauvegarde tame: %s\n", saveFile.c_str());
    printf("  Arret          : Ctrl+C\n"
           "==============================================================\n\n");

    signal(SIGINT, on_sigint); signal(SIGTERM, on_sigint);
    int nVoices = max(1, g_nthreads + nGpu);
    g_cnt = new Counter[nVoices]; g_tcnt = new Counter[nVoices];
    for (int i = 0; i < nVoices; i++) { g_cnt[i].v = 0; g_tcnt[i].v = 0; }
    vector<thread> th;
    auto t0 = chrono::steady_clock::now();
    for (int i = 0; i < g_nthreads; i++) th.emplace_back(cpu_worker, i, g_batch);
    for (int g = 0; g < nGpu; g++) th.emplace_back(gpu_worker, g, g_nthreads + g);

    double lastP = 0, lastSave = 0; u64 lastSteps = 0; bool endReached = false;
    while (!g_stop.load()) {
        this_thread::sleep_for(chrono::milliseconds(100));
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        if (duree && el >= duree) { g_stop = 1; break; }
        if (tameGoal) {
            u64 t2 = 0; for (int i = 0; i < nVoices; i++) t2 += g_cnt[i].v.load();
            if ((double)t2 + baseSteps >= endTarget) { endReached = true; g_stop = 1; break; }
        }
        if (mixGoal && !g_tameOff.load()) {
            u64 tt = 0; for (int i = 0; i < nVoices; i++) tt += g_tcnt[i].v.load();
            if ((double)tt + baseSteps >= endTarget) {
                g_tameOff = 1; endReached = true;
                printf("---- %s : cible tame atteinte -> tous les kangourous passent en wild\n\n", fmt_time(el).c_str());
                if (!saveFile.empty()) { save_tame(saveFile); lastSave = el; }
            }
        }
        if (el - lastP >= tsec) {
            u64 tot = 0; for (int i = 0; i < nVoices; i++) tot += g_cnt[i].v.load();
            double rate = (double)(tot - lastSteps) / (el - lastP);
            char eta[96] = "-", prog[48] = "-";
            if (g_nk > 0 || tameGoal) {
                snprintf(prog, sizeof prog, tameGoal ? "%.0f %% de la cible tame" : "%.0f %% de la moyenne attendue", 100.0 * ((double)tot + baseSteps) / Nexp);
                if ((double)tot + baseSteps < Nexp && rate > 0) snprintf(eta, sizeof eta, "~%s", fmt_time((Nexp - (double)tot - baseSteps) / rate).c_str());
                else snprintf(eta, sizeof eta, "moyenne depassee (variance normale)");
            }
            printf("---- %s ------------------------------------------\n"
                   "  Vitesse     : %.2f M sauts/s (GPU+CPU)\n"
                   "  Sauts       : %.3g  (%.2f x sqrt(W))\n"
                   "  Progression : %s\n"
                   "  Reste (moy.): %s\n"
                   "  DP stockes  : %llu  (tame : %llu)\n"
                   "  Cles        : %d / %d\n\n",
                   fmt_time(el).c_str(), rate / 1e6, (double)tot, (double)tot / sqW, prog, eta,
                   (unsigned long long)g_ndp.load(), (unsigned long long)g_ntame.load(), g_nsolved.load(), g_nk);
            lastP = el; lastSteps = tot;
        }
        if (!saveFile.empty() && g_doTame && !g_tameOff.load() && el - lastSave >= 300) { save_tame(saveFile); lastSave = el; }
    }
    g_stop = 1;
    for (auto& t : th) t.join();
    double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    u64 tot = 0; for (int i = 0; i < nVoices; i++) tot += g_cnt[i].v.load();
    if (!saveFile.empty() && g_doTame) {
        if (save_tame(saveFile)) printf("Table tame sauvegardee dans %s (%llu DP)\n", saveFile.c_str(), (unsigned long long)g_ntame.load());
    }
    printf("==============================================================\n"
           "  RESUME\n"
           "==============================================================\n"
           "  Duree          : %s\n"
           "  Sauts          : %.3g  (%.2f x sqrt(W))\n"
           "  Cles trouvees  : %d / %d\n", fmt_time(el).c_str(), (double)tot, (double)tot / sqW, g_nsolved.load(), g_nk);
    {
        const u64 nd = g_ndp.load(), nt = g_ntame.load();
        printf("  Points DP      : %d bits\n", g_dp);
        printf("  DP stockes     : %llu  (tame : %llu, wild : %llu)\n", (unsigned long long)nd, (unsigned long long)nt, (unsigned long long)(nd - nt));
        if (!saveFile.empty() && g_doTame) printf("  Fichier tame   : %s\n", saveFile.c_str());
        if (tameGoal) printf("  Cible tame     : %s\n", endReached ? "ATTEINTE" : "non atteinte");
    }
    if (g_nsolved.load() > 0) printf("  Fichier        : %s\n", g_outfile.c_str());
    printf("==============================================================\n");
    return g_nk > 0 && g_nsolved.load() == g_nk ? 0 : 2;
}
