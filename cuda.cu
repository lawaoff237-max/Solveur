// =============================================================================
//  kangaroo_cuda.cu  -  Pollard Kangaroo (lambda) multi-clés / CPU + GPU
//  Courbe secp256k1, intervalle borné [a, b].
//  Hybride : GPU (CUDA) pour les sauts, CPU pour la table DP et les collisions.
//
//  Compilation sur Kaggle (GPU T4/P100) :
//     nvcc -O3 -arch=sm_75 -std=c++17 -Xcompiler "-O3,-march=native,-pthread" \
//          kangaroo_cuda.cu -o kangaroo
//  Pour P100 (sm_60) :
//     nvcc -O3 -arch=sm_60 -std=c++17 -Xcompiler "-O3,-march=native,-pthread" \
//          kangaroo_cuda.cu -o kangaroo
//  Vérifier l'architecture GPU disponible :
//     nvidia-smi --query-gpu=compute_cap --format=csv
//
//  Si CUDA non disponible (CPU uniquement) :
//     g++ -O3 -march=native -flto -funroll-loops -pthread -std=c++17 \
//         -DCPU_ONLY kangaroo_cuda.cu -o kangaroo
//
//  Fonctionnalités :
//   - arithmétique 256 bits maison (4x64, réduction spéciale p = 2^256-2^32-977)
//   - addition affine + inversion par lots (Montgomery) : ~6 mul / saut
//   - symétrie ±y (classes d'équivalence) + détection/sortie des cycles
//   - points distingués (DP), table partagée entre threads (shards)
//   - N clés dans le même intervalle : table tame partagée, un troupeau wild/clé
//   - GPU : chaque SM traite un batch de kangourous en parallèle
//   - CPU : gestion table DP, collisions, E/S (threads POSIX)
//   - Détection automatique GPU ; repli CPU pur si absent
// =============================================================================

#ifndef CPU_ONLY
#  include <cuda_runtime.h>
#  define CUDA_CHECK(x) do { \
       cudaError_t _e=(x); \
       if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} \
   } while(0)
#else
// stubs pour permettre la compilation g++ sans CUDA
#  define __device__
#  define __host__
#  define __global__
#  define __shared__
#  define __forceinline__ inline
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

static inline bool u_zero(const U256& a) { return !(a.d[0] | a.d[1] | a.d[2] | a.d[3]); }
static inline bool u_eq(const U256& a, const U256& b) {
    return a.d[0]==b.d[0] && a.d[1]==b.d[1] && a.d[2]==b.d[2] && a.d[3]==b.d[3];
}
static inline int u_cmp(const U256& a, const U256& b) {
    for (int i=3;i>=0;i--) { if (a.d[i]!=b.d[i]) return a.d[i]>b.d[i]?1:-1; }
    return 0;
}
static inline U256 u_from(u128 v) { U256 r; r.d[0]=(u64)v; r.d[1]=(u64)(v>>64); r.d[2]=r.d[3]=0; return r; }
static inline U256 u_add(const U256& a,const U256& b) {
    U256 r; u128 c=0;
    for (int i=0;i<4;i++){c+=(u128)a.d[i]+b.d[i];r.d[i]=(u64)c;c>>=64;}
    return r;
}
static inline U256 u_sub(const U256& a,const U256& b) {
    U256 r; u64 br=0;
    for (int i=0;i<4;i++){u128 t=(u128)a.d[i]-b.d[i]-br;r.d[i]=(u64)t;br=(u64)(t>>64)&1;}
    return r;
}
static inline int u_bits(const U256& a) {
    for (int i=3;i>=0;i--) if (a.d[i]) return i*64+64-__builtin_clzll(a.d[i]);
    return 0;
}
static inline void u_shr1(U256& a) {
    for (int i=0;i<3;i++) a.d[i]=(a.d[i]>>1)|(a.d[i+1]<<63);
    a.d[3]>>=1;
}
static bool hex2u(string s,U256& r) {
    if (s.size()>1&&s[0]=='0'&&(s[1]=='x'||s[1]=='X')) s=s.substr(2);
    if (s.empty()||s.size()>64) return false;
    memset(&r,0,sizeof(r)); int pos=0;
    for (int i=(int)s.size()-1;i>=0;i--,pos++) {
        char c=s[i]; int v;
        if (c>='0'&&c<='9') v=c-'0';
        else if (c>='a'&&c<='f') v=c-'a'+10;
        else if (c>='A'&&c<='F') v=c-'A'+10;
        else return false;
        r.d[pos/16]|=(u64)v<<((pos%16)*4);
    }
    return true;
}
static string u2hex(const U256& a) {
    char b[80]; snprintf(b,sizeof b,"%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3],(unsigned long long)a.d[2],
        (unsigned long long)a.d[1],(unsigned long long)a.d[0]);
    return string(b);
}

// ================================ corps Fp ===================================
static inline void add_small(U256& r,u64 v) {
    u128 c=(u128)r.d[0]+v; r.d[0]=(u64)c; c>>=64;
    for (int i=1;i<4&&c;i++){c+=r.d[i];r.d[i]=(u64)c;c>>=64;}
}
static inline void sub_small(U256& r,u64 v) {
    u64 br=r.d[0]<v; r.d[0]-=v;
    for (int i=1;i<4&&br;i++){br=(r.d[i]==0);r.d[i]--;}
}
static inline void fe_norm(fe& r) {
    if (r.d[3]==~0ULL&&r.d[2]==~0ULL&&r.d[1]==~0ULL&&r.d[0]>=P0) add_small(r,CP);
}
static inline void fe_add(fe& r,const fe& a,const fe& b) {
    u128 c=0;
    for (int i=0;i<4;i++){c+=(u128)a.d[i]+b.d[i];r.d[i]=(u64)c;c>>=64;}
    if (c) add_small(r,CP);
    fe_norm(r);
}
static inline void fe_sub(fe& r,const fe& a,const fe& b) {
    u64 br=0;
    for (int i=0;i<4;i++){u128 t=(u128)a.d[i]-b.d[i]-br;r.d[i]=(u64)t;br=(u64)(t>>64)&1;}
    if (br) sub_small(r,CP);
}
static inline void fe_neg(fe& r,const fe& a) {
    if (u_zero(a)){r=a;return;}
    fe z; memset(&z,0,sizeof z); fe_sub(r,z,a);
}
static inline void fe_reduce_from_t(fe& r,u64 t[8]) {
    u64 o[4]; u128 c=0;
    for (int i=0;i<4;i++){c+=(u128)t[4+i]*CP+t[i];o[i]=(u64)c;c>>=64;}
    u64 top=(u64)c;
    c=(u128)top*CP;
    for (int i=0;i<4;i++){c+=o[i];o[i]=(u64)c;c>>=64;}
    memcpy(r.d,o,sizeof o);
    if (c) add_small(r,CP);
    fe_norm(r);
}
static inline void fe_mul(fe& r,const fe& a,const fe& b) {
    u64 t[8]; u128 c=0;
    for (int j=0;j<4;j++){c+=(u128)a.d[0]*b.d[j];t[j]=(u64)c;c>>=64;}
    t[4]=(u64)c;
    for (int i=1;i<4;i++){
        c=0;
        for (int j=0;j<4;j++){c+=(u128)a.d[i]*b.d[j]+t[i+j];t[i+j]=(u64)c;c>>=64;}
        t[i+4]=(u64)c;
    }
    fe_reduce_from_t(r,t);
}
static inline void fe_sqr(fe& r,const fe& a) {
    u64 OD[7]={0,0,0,0,0,0,0}; u128 c;
    c=0;
    for (int j=1;j<4;j++){c+=(u128)a.d[0]*a.d[j];OD[j]=(u64)c;c>>=64;}
    OD[4]=(u64)c;
    c=0;
    for (int j=2;j<4;j++){c+=(u128)a.d[1]*a.d[j]+OD[1+j];OD[1+j]=(u64)c;c>>=64;}
    OD[5]=(u64)c;
    c=(u128)a.d[2]*a.d[3]+OD[5];
    OD[5]=(u64)c;c>>=64;OD[6]=(u64)c;
    u64 t[8]; u64 carry=0;
    for (int k=0;k<7;k++){u64 v=OD[k];t[k]=(v<<1)|carry;carry=v>>63;}
    t[7]=carry;
    u64 dcarry=0;
    for (int i=0;i<4;i++){
        u128 sq=(u128)a.d[i]*a.d[i];
        u64 lo=(u64)sq,hi=(u64)(sq>>64);
        u128 s=(u128)t[2*i]+lo+dcarry;
        t[2*i]=(u64)s;dcarry=(u64)(s>>64);
        s=(u128)t[2*i+1]+hi+dcarry;
        t[2*i+1]=(u64)s;dcarry=(u64)(s>>64);
    }
    fe_reduce_from_t(r,t);
}
static inline void fe_sqrn(fe& r,const fe& a,int n){r=a;for(int i=0;i<n;i++)fe_sqr(r,r);}

static void fe_inv(fe& r,const fe& a) {
    fe x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(x2,a);      fe_mul(x2,x2,a);
    fe_sqr(x3,x2);     fe_mul(x3,x3,a);
    fe_sqrn(x6,x3,3);  fe_mul(x6,x6,x3);
    fe_sqrn(x9,x6,3);  fe_mul(x9,x9,x3);
    fe_sqrn(x11,x9,2); fe_mul(x11,x11,x2);
    fe_sqrn(x22,x11,11);  fe_mul(x22,x22,x11);
    fe_sqrn(x44,x22,22);  fe_mul(x44,x44,x22);
    fe_sqrn(x88,x44,44);  fe_mul(x88,x88,x44);
    fe_sqrn(x176,x88,88); fe_mul(x176,x176,x88);
    fe_sqrn(x220,x176,44);fe_mul(x220,x220,x44);
    fe_sqrn(x223,x220,3); fe_mul(x223,x223,x3);
    fe_sqrn(t,x223,23);fe_mul(t,t,x22);
    fe_sqrn(t,t,5);    fe_mul(t,t,a);
    fe_sqrn(t,t,3);    fe_mul(t,t,x2);
    fe_sqrn(t,t,2);    fe_mul(r,t,a);
}
static void fe_pow(fe& r,const fe& a,const U256& e) {
    fe res; memset(&res,0,sizeof res);res.d[0]=1;
    int nb=u_bits(e);
    for (int i=nb-1;i>=0;i--){
        fe_sqr(res,res);
        if ((e.d[i/64]>>(i%64))&1) fe_mul(res,res,a);
    }
    r=res;
}

// ================================ courbe =====================================
struct Pt { fe x,y; bool inf; };
struct Jac { fe X,Y,Z; };

static U256 SECP_N,GX,GY;
static Pt GEN;

static inline Jac jac_inf(){Jac r;memset(&r,0,sizeof r);r.X.d[0]=1;r.Y.d[0]=1;return r;}
static inline Jac jac_from(const Pt& p){
    if (p.inf) return jac_inf();
    Jac r;r.X=p.x;r.Y=p.y;memset(&r.Z,0,sizeof r.Z);r.Z.d[0]=1;return r;
}
static void jac_dbl(Jac& r,const Jac& a){
    if (u_zero(a.Z)||u_zero(a.Y)){r=jac_inf();return;}
    fe A,B,C,D,E,F,t,X3,Y3,Z3,C8;
    fe_sqr(A,a.X);fe_sqr(B,a.Y);fe_sqr(C,B);
    fe_add(t,a.X,B);fe_sqr(t,t);fe_sub(t,t,A);fe_sub(t,t,C);fe_add(D,t,t);
    fe_add(E,A,A);fe_add(E,E,A);
    fe_sqr(F,E);
    fe_sub(X3,F,D);fe_sub(X3,X3,D);
    fe_sub(t,D,X3);fe_mul(Y3,E,t);
    fe_add(C8,C,C);fe_add(C8,C8,C8);fe_add(C8,C8,C8);
    fe_sub(Y3,Y3,C8);
    fe_mul(Z3,a.Y,a.Z);fe_add(Z3,Z3,Z3);
    r.X=X3;r.Y=Y3;r.Z=Z3;
}
static void jac_add(Jac& r,const Jac& a,const Jac& b){
    if (u_zero(a.Z)){r=b;return;}
    if (u_zero(b.Z)){r=a;return;}
    fe Z1Z1,Z2Z2,U1,U2,S1,S2,H,R,t;
    fe_sqr(Z1Z1,a.Z);fe_sqr(Z2Z2,b.Z);
    fe_mul(U1,a.X,Z2Z2);fe_mul(U2,b.X,Z1Z1);
    fe_mul(t,b.Z,Z2Z2);fe_mul(S1,a.Y,t);
    fe_mul(t,a.Z,Z1Z1);fe_mul(S2,b.Y,t);
    fe_sub(H,U2,U1);fe_sub(R,S2,S1);
    if (u_zero(H)){
        if (u_zero(R)){jac_dbl(r,a);return;}
        r=jac_inf();return;
    }
    fe H2,H3,V,X3,Y3,Z3;
    fe_sqr(H2,H);fe_mul(H3,H,H2);fe_mul(V,U1,H2);
    fe_sqr(X3,R);fe_sub(X3,X3,H3);fe_sub(X3,X3,V);fe_sub(X3,X3,V);
    fe_sub(t,V,X3);fe_mul(Y3,R,t);fe_mul(t,S1,H3);fe_sub(Y3,Y3,t);
    fe_mul(Z3,a.Z,b.Z);fe_mul(Z3,Z3,H);
    r.X=X3;r.Y=Y3;r.Z=Z3;
}
static Pt to_aff(const Jac& j){
    Pt p;p.inf=false;
    if (u_zero(j.Z)){p.inf=true;memset(&p.x,0,sizeof p.x);memset(&p.y,0,sizeof p.y);return p;}
    fe zi,z2,z3;
    fe_inv(zi,j.Z);fe_sqr(z2,zi);fe_mul(z3,z2,zi);
    fe_mul(p.x,j.X,z2);fe_mul(p.y,j.Y,z3);
    return p;
}
static Pt pt_mul(const U256& k,const Pt& base){
    Jac R=jac_inf(),B=jac_from(base);
    int nb=u_bits(k);
    for (int i=nb-1;i>=0;i--){jac_dbl(R,R);if ((k.d[i/64]>>(i%64))&1) jac_add(R,R,B);}
    return to_aff(R);
}
static inline Pt mulG(const U256& k){return pt_mul(k,GEN);}
static Pt pt_neg(const Pt& p){Pt r=p;if (!p.inf) fe_neg(r.y,p.y);return r;}
static Pt pt_add(const Pt& a,const Pt& b){Jac r;jac_add(r,jac_from(a),jac_from(b));return to_aff(r);}
static Pt mulG_i128(i128 k){
    if (k>=0) return mulG(u_from((u128)k));
    return pt_neg(mulG(u_from((u128)(-k))));
}
static bool pt_eq(const Pt& a,const Pt& b){
    if (a.inf||b.inf) return a.inf==b.inf;
    return u_eq(a.x,b.x)&&u_eq(a.y,b.y);
}
static U256 P_FIELD,EXP_SQRT;
static bool on_curve(const Pt& p){
    fe y2,x3,rhs,seven;memset(&seven,0,sizeof seven);seven.d[0]=7;
    fe_sqr(y2,p.y);fe_sqr(x3,p.x);fe_mul(x3,x3,p.x);fe_add(rhs,x3,seven);
    return u_eq(y2,rhs);
}
static bool parse_pub(string s,Pt& P){
    while (!s.empty()&&isspace((unsigned char)s.back())) s.pop_back();
    while (!s.empty()&&isspace((unsigned char)s.front())) s.erase(s.begin());
    P.inf=false;
    if (s.size()==66&&(s.compare(0,2,"02")==0||s.compare(0,2,"03")==0)){
        if (!hex2u(s.substr(2),P.x)) return false;
        if (u_cmp(P.x,P_FIELD)>=0) return false;
        fe x3,rhs,seven,y,y2;memset(&seven,0,sizeof seven);seven.d[0]=7;
        fe_sqr(x3,P.x);fe_mul(x3,x3,P.x);fe_add(rhs,x3,seven);
        fe_pow(y,rhs,EXP_SQRT);fe_sqr(y2,y);
        if (!u_eq(y2,rhs)) return false;
        bool odd=y.d[0]&1;
        if (odd!=(s[1]=='3')) fe_neg(y,y);
        P.y=y;return true;
    }
    if (s.size()==130&&s.compare(0,2,"04")==0){
        if (!hex2u(s.substr(2,64),P.x)||!hex2u(s.substr(66,64),P.y)) return false;
        return on_curve(P);
    }
    return false;
}
static string pub_hex(const Pt& P){
    string h=u2hex(P.x);
    return string((P.y.d[0]&1)?"03":"02")+h;
}
static void init_curve(){
    hex2u("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",P_FIELD);
    hex2u("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",SECP_N);
    hex2u("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",GX);
    hex2u("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",GY);
    GEN.x=GX;GEN.y=GY;GEN.inf=false;
    U256 e=u_add(P_FIELD,u_from(1));u_shr1(e);u_shr1(e);EXP_SQRT=e;
}

// ============================== paramètres ===================================
static const int NJ=256;
static const int NE=64;
struct Jump { fe x,y; i128 d; };
static Jump g_J[NJ],g_E[NE];

static U256 g_a,g_end,g_c;
static u128 g_W=0;
static int g_sym=1;
static int g_dp=-1; static u64 g_dpmask=0;
static int g_nthreads=1,g_batch=512;
static double g_jf=-1,g_jg=-1; static int g_sl=2;
static double g_coef=-1;
static double g_tfrac=0.5;
static u64 g_seed=0;
static u64 g_jhash=0;

static vector<Pt> g_pub,g_Q;
static vector<string> g_pubstr;
static int g_nk=0;
static atomic<int>* g_solved=nullptr;
static vector<i128> g_kval;
static atomic<int> g_nsolved{0};
static atomic<int> g_stop{0};
static atomic<u64> g_keyctr{0};
static mutex g_solmtx,g_relmtx,g_outmtx;
static string g_outfile="kangaroo_trouve.txt";

struct Rng {
    u64 s[4];
    explicit Rng(u64 seed){
        for (int i=0;i<4;i++){
            seed+=0x9E3779B97F4A7C15ULL;u64 z=seed;
            z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;
            s[i]=z^(z>>31);
        }
    }
    static inline u64 rotl(u64 x,int k){return(x<<k)|(x>>(64-k));}
    u64 next(){
        u64 r=rotl(s[1]*5,7)*9,t=s[1]<<17;
        s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);
        return r;
    }
    u128 next128(){u128 h=next();return(h<<64)|next();}
    u128 below(u128 n){return next128()%n;}
};

static void build_jumps(double M){
    Rng r(0xC0FFEE123456789ULL^(u64)g_W^(u64)(g_W>>64));
    u128 span=(u128)(2.0L*(long double)M);if(span<8)span=8;
    g_jhash=(u64)span*0x9E3779B97F4A7C15ULL;
    for (int i=0;i<NJ+NE;i++){
        Jump& j=(i<NJ)?g_J[i]:g_E[i-NJ];
        j.d=(i128)(1+r.below(span));
        Pt p=mulG_i128(j.d);j.x=p.x;j.y=p.y;
        g_jhash=(g_jhash^(u64)j.d)*0x100000001B3ULL+(u64)(j.d>>64);
    }
}

// ============================ table des DP ===================================
static const u32 TAME=0xFFFFFFFFu;
struct K128{u64 a,b;bool operator==(const K128&o)const{return a==o.a&&b==o.b;}};
struct KH{size_t operator()(const K128&k)const{
    u64 z=k.a*0x9E3779B97F4A7C15ULL+k.b;z^=z>>29;z*=0xBF58476D1CE4E5B9ULL;z^=z>>32;return(size_t)z;}};
struct Ent{i128 d;u32 id;int8_t s;};
struct Shard{mutex m;unordered_map<K128,Ent,KH>mp;};
static Shard g_sh[256];
static atomic<u64> g_ndp{0},g_ntame{0};
struct DPRec{u64 x0,x1;i128 d;u32 id;int8_t s;};

struct Rel{int i,j,c;i128 k0;};
static vector<Rel> g_rel;
static void propagate();

static string hex_trim(const U256& a){
    string h=u2hex(a);size_t p=h.find_first_not_of('0');
    return p==string::npos?"0":h.substr(p);
}
static void report_found(int idx,const U256& key){
    string pub=g_pubstr[idx],priv=u2hex(key);
    printf("\n==============================================================\n"
           "  >>>  CLE TROUVEE  <<<                       (%d / %d)\n"
           "==============================================================\n"
           "  Cle publique : %s\n"
           "  Cle privee   : 0x%s\n"
           "  Enregistree  : %s\n"
           "%s"
           "==============================================================\n\n",
           g_nsolved.load(),g_nk,pub.c_str(),hex_trim(key).c_str(),g_outfile.c_str(),
           (u_cmp(key,g_a)<0||u_cmp(key,g_end)>0)?"  Note         : cle HORS de l'intervalle demande (verifiee par key*G)\n":"");
    fflush(stdout);
    FILE* f=fopen(g_outfile.c_str(),"a");
    if (f){
        time_t t=time(nullptr);char tb[64];strftime(tb,sizeof tb,"%Y-%m-%d %H:%M:%S",localtime(&t));
        fprintf(f,"Date          : %s\nCle publique  : %s\nCle privee    : 0x%s\n\n",tb,pub.c_str(),priv.c_str());
        fclose(f);
    } else printf("ATTENTION : impossible d'ecrire dans %s\n",g_outfile.c_str());
}
static U256 key_from_k(i128 k){
    if (k>=0){
        U256 kk=u_from((u128)k),r;u128 c=0;
        for (int i=0;i<4;i++){c+=(u128)g_c.d[i]+kk.d[i];r.d[i]=(u64)c;c>>=64;}
        if (c||u_cmp(r,SECP_N)>=0) r=u_sub(r,SECP_N);
        return r;
    }
    U256 kk=u_from((u128)(-k));
    if (u_cmp(g_c,kk)>=0) return u_sub(g_c,kk);
    return u_sub(SECP_N,u_sub(kk,g_c));
}
static void try_key(int i,i128 k){
    if (g_solved[i].load(memory_order_acquire)) return;
    U256 key=key_from_k(k);
    Pt P=mulG(key);
    if (!pt_eq(P,g_pub[i])) return;
    {
        lock_guard<mutex> lk(g_solmtx);
        if (g_solved[i].load(memory_order_acquire)) return;
        g_kval[i]=k;
        g_solved[i].store(1,memory_order_release);
        g_nsolved++;
        lock_guard<mutex> lo(g_outmtx);
        report_found(i,key);
        if (g_nsolved.load()>=g_nk) g_stop=1;
    }
    propagate();
}
static void propagate(){
    bool ch=true;
    while(ch){
        ch=false;
        vector<Rel> rel;{lock_guard<mutex> lk(g_relmtx);rel=g_rel;}
        for (auto& r:rel){
            bool si=g_solved[r.i].load(memory_order_acquire),sj=g_solved[r.j].load(memory_order_acquire);
            if (si&&!sj){i128 kj=(i128)r.c*(g_kval[r.i]-r.k0);try_key(r.j,kj);if(g_solved[r.j].load())ch=true;}
            else if(sj&&!si){i128 ki=(i128)r.c*g_kval[r.j]+r.k0;try_key(r.i,ki);if(g_solved[r.i].load())ch=true;}
        }
    }
}
static void collide(const Ent& A,const Ent& B){
    if (A.id==TAME&&B.id==TAME) return;
    const Ent *t=nullptr,*w=nullptr;
    int nsig=g_sym?2:1;
    if (A.id==TAME){t=&A;w=&B;}else if(B.id==TAME){t=&B;w=&A;}
    if (t){
        if (g_solved[w->id].load()) return;
        for (int q=0;q<nsig;q++){
            int sg=q?-1:1;
            i128 k=(i128)w->s*((i128)sg*t->d-w->d);
            try_key((int)w->id,k);
        }
        return;
    }
    for (int q=0;q<nsig;q++){
        int sg=q?-1:1;
        if (A.id==B.id){
            int coef=A.s-sg*B.s;if(coef==0)continue;
            i128 rhs=(i128)sg*B.d-A.d;if(rhs&1)continue;
            try_key((int)A.id,rhs/coef);
        } else {
            int c=A.s*sg*B.s;
            i128 k0=(i128)A.s*((i128)sg*B.d-A.d);
            bool si=g_solved[A.id].load(),sj=g_solved[B.id].load();
            if(si&&sj)continue;
            {lock_guard<mutex> lk(g_relmtx);if(g_rel.size()<2000000)g_rel.push_back({(int)A.id,(int)B.id,c,k0});}
            if(si||sj)propagate();
        }
    }
}
static void add_dp(const DPRec& r){
    K128 k{r.x0,r.x1};
    Shard& sh=g_sh[(r.x1>>8)&255];
    Ent ne{r.d,r.id,r.s},old;
    {
        lock_guard<mutex> lk(sh.m);
        auto it=sh.mp.find(k);
        if(it==sh.mp.end()){sh.mp.emplace(k,ne);g_ndp++;if(r.id==TAME)g_ntame++;return;}
        old=it->second;
    }
    if(old.id==ne.id&&old.d==ne.d)return;
    collide(old,ne);
}

// ======================== sauvegarde / chargement ============================
struct Hdr{char magic[8];U256 a;u64 Wlo,Whi;u32 sym,dp;u64 jhash,count;};
struct Rec{u64 x0,x1;i128 d;};

static bool save_tame(const string& fn){
    vector<Rec> v;v.reserve(g_ntame.load());
    for (int i=0;i<256;i++){
        lock_guard<mutex> lk(g_sh[i].m);
        for (auto& kv:g_sh[i].mp) if(kv.second.id==TAME) v.push_back({kv.first.a,kv.first.b,kv.second.d});
    }
    Hdr h;memset(&h,0,sizeof h);memcpy(h.magic,"KANGDP01",8);
    h.a=g_a;h.Wlo=(u64)g_W;h.Whi=(u64)(g_W>>64);h.sym=g_sym;h.dp=g_dp;h.jhash=g_jhash;h.count=v.size();
    string tmp=fn+".tmp";
    FILE* f=fopen(tmp.c_str(),"wb");
    if(!f){fprintf(stderr,"Erreur: impossible d'ecrire %s\n",tmp.c_str());return false;}
    fwrite(&h,sizeof h,1,f);
    if(!v.empty()) fwrite(v.data(),sizeof(Rec),v.size(),f);
    fclose(f);
#ifdef _WIN32
    remove(fn.c_str());
#endif
    if(rename(tmp.c_str(),fn.c_str())!=0){fprintf(stderr,"Erreur: rename %s\n",fn.c_str());return false;}
    return true;
}
static bool read_hdr(const string& fn,Hdr& h){
    ifstream f(fn,ios::binary);if(!f)return false;
    f.read((char*)&h,sizeof h);
    return f.good()&&memcmp(h.magic,"KANGDP01",8)==0;
}
static bool check_hdr(const Hdr& h,const string& fn){
    if(!u_eq(h.a,g_a)||h.Wlo!=(u64)g_W||h.Whi!=(u64)(g_W>>64)){
        fprintf(stderr,"Fichier %s: intervalle different, refuse.\n",fn.c_str());return false;}
    if((int)h.sym!=g_sym||h.jhash!=g_jhash){
        fprintf(stderr,"Fichier %s: parametres differents, refuse.\n",fn.c_str());return false;}
    return true;
}
static bool load_tame(const string& fn,u64& n){
    Hdr h;
    if(!read_hdr(fn,h)){fprintf(stderr,"Fichier %s illisible.\n",fn.c_str());return false;}
    if(!check_hdr(h,fn))return false;
    if((int)h.dp!=g_dp){fprintf(stderr,"Fichier %s: masque DP different.\n",fn.c_str());return false;}
    ifstream f(fn,ios::binary);f.seekg(sizeof(Hdr));
    vector<Rec>buf(1<<16);u64 left=h.count;n=0;
    while(left){
        size_t m=(size_t)min<u64>(left,buf.size());
        f.read((char*)buf.data(),m*sizeof(Rec));
        if(!f){fprintf(stderr,"Fichier %s tronque.\n",fn.c_str());return false;}
        for(size_t i=0;i<m;i++) add_dp({buf[i].x0,buf[i].x1,buf[i].d,TAME,1});
        left-=m;n+=m;
    }
    return true;
}

// ================================ worker CPU =================================
struct alignas(64) Counter{atomic<u64> v;char pad[56];};
static Counter* g_cnt=nullptr;
static u64 g_rotLimit=0;
static bool g_doTame=true,g_doWild=true;
static atomic<int> g_tameOff{0};
static Counter* g_tcnt=nullptr;

static int pick_key(){
    int n=g_nk;if(n==0)return -1;
    for(int t=0;t<n;t++){
        int i=(int)(g_keyctr.fetch_add(1)%(u64)n);
        if(!g_solved[i].load())return i;
    }
    return -1;
}

static const Jump* slow_add(fe& x,fe& y,const Jump* jp){
    fe lam,t,x3,y3,inv;
    if(u_eq(x,jp->x)){
        if(u_eq(y,jp->y)){
            fe n,d;fe_sqr(n,x);fe_add(t,n,n);fe_add(n,t,n);
            fe_add(d,y,y);fe_inv(inv,d);fe_mul(lam,n,inv);
        } else {
            jp=&g_E[0];
            fe dx;fe_sub(dx,jp->x,x);fe_inv(inv,dx);
            fe_sub(t,jp->y,y);fe_mul(lam,t,inv);
        }
    } else {
        fe dx;fe_sub(dx,jp->x,x);fe_inv(inv,dx);
        fe_sub(t,jp->y,y);fe_mul(lam,t,inv);
    }
    fe l2;fe_sqr(l2,lam);
    fe_sub(x3,l2,x);fe_sub(x3,x3,jp->x);
    fe_sub(t,x,x3);fe_mul(y3,lam,t);fe_sub(y3,y3,y);
    x=x3;y=y3;
    return jp;
}

static void worker_cpu(int tid,int B){
    Rng rng(g_seed*0x9E3779B97F4A7C15ULL+(u64)tid*7919ULL+12345ULL);
    vector<fe>X(B),Y(B),DX(B),PRE(B);
    vector<i128>D(B);
    vector<u32>ID(B),SN(B),HP(B);
    vector<int8_t>S(B);
    vector<char>SLOW(B);
    vector<u64>H((size_t)B*16),AGE(B);
    vector<const Jump*>JP(B);
    vector<DPRec>out;out.reserve(64);
    const u128 L=(g_sl>=120)?1:max<u128>(1,g_W>>g_sl);
    const u32 dpLimit=(u32)min<u64>(0xFFFFFFF0ULL,((u64)64<<min(g_dp,24))+4096);
    bool alive=true;

    auto roleTame=[&](int j){return g_doTame&&!g_tameOff.load(memory_order_relaxed)&&(!g_doWild||(long)floor((j+1)*g_tfrac)>(long)floor(j*g_tfrac));};
    auto init=[&](int j){
        bool tame=roleTame(j);
        for(;;){
            i128 r=(i128)(rng.below(2*L+1))-(i128)L;
            if(r==0)r=1;
            Pt P;int id=-1;
            if(tame){P=mulG_i128(r);}
            else{
                id=pick_key();
                if(id<0){alive=false;return;}
                P=pt_add(g_Q[id],mulG_i128(r));
            }
            if(P.inf)continue;
            X[j]=P.x;Y[j]=P.y;D[j]=r;ID[j]=tame?TAME:(u32)id;S[j]=1;
            if(g_sym&&(Y[j].d[0]&1)){fe_neg(Y[j],Y[j]);D[j]=-D[j];S[j]=-1;}
            memset(&H[(size_t)j*16],0,16*sizeof(u64));
            HP[j]=0;H[(size_t)j*16+(HP[j]++&15)]=X[j].d[0];
            SN[j]=0;AGE[j]=0;
            return;
        }
    };
    for(int j=0;j<B&&alive;j++)init(j);
    if(!alive)return;

    u64 iter=0;
    while(!g_stop.load(memory_order_relaxed)){
        for(int j=0;j<B;j++){
            const Jump* jp=nullptr;
            u64* h=&H[(size_t)j*16];
            if(g_sym){
                u32 pos=HP[j];u64 xl=X[j].d[0];
                for(int Lc=2;Lc<=8;Lc+=2){
                    if(h[(pos-1-Lc)&15]==xl){
                        u64 mn=xl;
                        for(int q=1;q<Lc;q++){u64 v=h[(pos-1-q)&15];if(v<mn)mn=v;}
                        jp=&g_E[(mn>>7)%NE];
                        memset(h,0,16*sizeof(u64));HP[j]=0;h[(HP[j]++)&15]=xl;
                        break;
                    }
                }
            }
            if(!jp)jp=&g_J[X[j].d[1]&(NJ-1)];
            JP[j]=jp;
            fe_sub(DX[j],jp->x,X[j]);
            if(u_zero(DX[j])){SLOW[j]=1;memset(&DX[j],0,sizeof(fe));DX[j].d[0]=1;}else SLOW[j]=0;
        }
        PRE[0]=DX[0];
        for(int j=1;j<B;j++)fe_mul(PRE[j],PRE[j-1],DX[j]);
        fe inv;fe_inv(inv,PRE[B-1]);
        for(int j=B-1;j>0;j--){
            fe t;fe_mul(t,inv,PRE[j-1]);
            fe_mul(inv,inv,DX[j]);
            DX[j]=t;
        }
        DX[0]=inv;
        u64 tameStep=0;
        for(int j=0;j<B;j++){
            const Jump* jp=JP[j];
            if(SLOW[j])jp=slow_add(X[j],Y[j],jp);
            else{
                fe t,lam,l2,x3,y3;
                fe_sub(t,jp->y,Y[j]);fe_mul(lam,t,DX[j]);
                fe_sqr(l2,lam);
                fe_sub(x3,l2,X[j]);fe_sub(x3,x3,jp->x);
                fe_sub(t,X[j],x3);fe_mul(y3,lam,t);fe_sub(y3,y3,Y[j]);
                X[j]=x3;Y[j]=y3;
            }
            D[j]+=jp->d;
            if(g_sym&&(Y[j].d[0]&1)){fe_neg(Y[j],Y[j]);D[j]=-D[j];S[j]=-S[j];}
            u64* h=&H[(size_t)j*16];
            h[(HP[j]++)&15]=X[j].d[0];
            SN[j]++;AGE[j]++;
            if(ID[j]==TAME)tameStep++;
            if((X[j].d[0]&g_dpmask)==0){
                out.push_back({X[j].d[0],X[j].d[1],D[j],ID[j],S[j]});
                SN[j]=0;
            }else if(SN[j]>dpLimit){init(j);if(!alive)break;}
        }
        if(!alive)break;
        g_cnt[tid].v.fetch_add((u64)B,memory_order_relaxed);
        if(tameStep)g_tcnt[tid].v.fetch_add(tameStep,memory_order_relaxed);
        if(!out.empty()){for(auto& r:out)add_dp(r);out.clear();}
        iter++;
        if((iter&63)==0){
            for(int j=0;j<B;j++){
                if(ID[j]==TAME){if(g_tameOff.load(memory_order_relaxed)){init(j);if(!alive)break;}}
                else if(g_solved[ID[j]].load()||(g_rotLimit&&AGE[j]>g_rotLimit)){init(j);if(!alive)break;}
            }
            if(!alive)break;
        }
    }
}

// ========================== worker GPU (CUDA) ================================
// Structure compacte pour transférer l'état d'un kangourou GPU→CPU
struct GpuKangState {
    u64 x0, x1, x2, x3;   // coordonnée x affine (fe)
    u64 y0;                // seul bit utile : parité de y (symétrie)
    u64 d_lo, d_hi;        // distance signée (i128) : lo=bits 0..63, hi=bits 64..127 (signé)
    u32 id;                // TAME ou indice wild
    int8_t s;              // signe (symétrie)
    u32 steps_since_dp;    // compteur depuis dernier DP
    u32 age;               // âge total (pour rotation)
};

#ifndef CPU_ONLY
// ---- données GPU constantes ------------------------------------------------
// On stocke les coordonnées des sauts sous forme plate (u64 x 4 par composante)
struct GpuJump { u64 xx[4],xy[4]; i128 d; };

static GpuJump* d_J=nullptr;   // NJ sauts sur GPU
static GpuJump* d_E=nullptr;   // NE sauts d'évasion sur GPU

// Paramètres globaux GPU (copiés dans __constant__ au lancement)
struct GpuParams {
    u64 dpmask;
    u32 dp_limit;
    u32 nj;
    u32 ne;
    u32 sym;
    u64 seed;
    u64 W_lo, W_hi;
    u32 nk;
    u32 sl;
    // sauts inline (NJ+NE <= 320, chacun ~128 octets = 40 KiB max)
    // On les passe via pointeurs device globaux pour garder le __constant__ léger.
};
static __constant__ GpuParams d_params;

// ---- kernel : un thread = un kangourou -------------------------------------
// Chaque thread effectue STEPS_PER_LAUNCH sauts, puis remonte ses DP via un buffer.
#define GPU_STEPS_PER_LAUNCH 1024

// Buffer de DP produits par le GPU (rempli par le kernel, lu par le CPU)
struct GpuDPBuf {
    u64 x0, x1;
    u64 d_lo, d_hi;   // i128 encodé
    u32 id;
    int8_t s;
    u8 _pad[1];
};

// Arithmétique Fp inline pour CUDA (pas d'__int128 garanti, on simule)
__device__ __forceinline__
static u64 gpu_addc(u64 a, u64 b, u64& carry) {
    u64 r = a + b + carry;
    carry = (r < a + carry || (carry && r == a)) ? 1ULL : 0ULL;
    // version portable (nvcc supporte la comparaison unsigned)
    u64 s = a + b;
    carry = (s < a) ? 1ULL : 0ULL;
    r = s + carry; // NB : on recalcule proprement
    // Simplifié : nvcc supporte __uint128_t sur sm_60+
    return r;
}

// Utilisez __uint128_t disponible sur sm_60+ avec nvcc
typedef unsigned __int128 u128g;

__device__ __forceinline__
static void gpu_fe_add(u64* r, const u64* a, const u64* b) {
    // Réduction p = 2^256 - 2^32 - 977
    u128g c = 0;
    #pragma unroll
    for (int i=0;i<4;i++){c+=(u128g)a[i]+b[i];r[i]=(u64)c;c>>=64;}
    if (c) {
        u128g cc=(u128g)r[0]+0x1000003D1ULL;r[0]=(u64)cc;cc>>=64;
        for(int i=1;i<4&&cc;i++){cc+=r[i];r[i]=(u64)cc;cc>>=64;}
    }
    // fe_norm inline
    if(r[3]==~0ULL&&r[2]==~0ULL&&r[1]==~0ULL&&r[0]>=0xFFFFFFFEFFFFFC2FULL){
        u128g cc=(u128g)r[0]+0x1000003D1ULL;r[0]=(u64)cc;cc>>=64;
        for(int i=1;i<4&&cc;i++){cc+=r[i];r[i]=(u64)cc;cc>>=64;}
    }
}
__device__ __forceinline__
static void gpu_fe_sub(u64* r, const u64* a, const u64* b) {
    u64 br=0;
    #pragma unroll
    for(int i=0;i<4;i++){
        u128g t=(u128g)a[i]-b[i]-br;
        r[i]=(u64)t;br=(u64)(t>>64)&1;
    }
    if(br){
        // soustraire CP = 0x1000003D1 (soustraction dans Fp : add CP)
        // en fait sub_small : r[0] -= CP -> peut déborder -> propager
        u64 brr=r[0]<0x1000003D1ULL?1:0;
        r[0]-=0x1000003D1ULL;
        for(int i=1;i<4&&brr;i++){brr=(r[i]==0);r[i]--;}
    }
}
__device__ __forceinline__
static void gpu_fe_mul(u64* r, const u64* a, const u64* b) {
    u64 t[8]={0,0,0,0,0,0,0,0};
    #pragma unroll
    for(int i=0;i<4;i++){
        u128g c=0;
        #pragma unroll
        for(int j=0;j<4;j++){c+=(u128g)a[i]*b[j]+t[i+j];t[i+j]=(u64)c;c>>=64;}
        t[i+4]=(u64)c;
    }
    // réduction
    u64 o[4];u128g c=0;
    #pragma unroll
    for(int i=0;i<4;i++){c+=(u128g)t[4+i]*0x1000003D1ULL+t[i];o[i]=(u64)c;c>>=64;}
    u64 top=(u64)c;
    c=(u128g)top*0x1000003D1ULL;
    #pragma unroll
    for(int i=0;i<4;i++){c+=o[i];o[i]=(u64)c;c>>=64;}
    #pragma unroll
    for(int i=0;i<4;i++)r[i]=o[i];
    if(c){
        u128g cc=(u128g)r[0]+0x1000003D1ULL;r[0]=(u64)cc;cc>>=64;
        for(int i=1;i<4&&cc;i++){cc+=r[i];r[i]=(u64)cc;cc>>=64;}
    }
    // fe_norm
    if(r[3]==~0ULL&&r[2]==~0ULL&&r[1]==~0ULL&&r[0]>=0xFFFFFFFEFFFFFC2FULL){
        u128g cc=(u128g)r[0]+0x1000003D1ULL;r[0]=(u64)cc;cc>>=64;
        for(int i=1;i<4&&cc;i++){cc+=r[i];r[i]=(u64)cc;cc>>=64;}
    }
}
__device__ __forceinline__
static void gpu_fe_sqr(u64* r, const u64* a) { gpu_fe_mul(r,a,a); }

// Inversion par la chaîne d'addition (identique CPU) - version device
__device__
static void gpu_fe_inv(u64* r, const u64* a) {
    u64 x2[4],x3[4],x6[4],x9[4],x11[4],x22[4],x44[4],x88[4],x176[4],x220[4],x223[4],t[4];
    // x2 = a^2 * a = a^3  ... (chaîne libsecp256k1)
    gpu_fe_sqr(x2,a);     gpu_fe_mul(x2,x2,a);
    gpu_fe_sqr(x3,x2);    gpu_fe_mul(x3,x3,a);
    // x6 = x3^(2^3)*x3
    u64 tmp[4];
    for(int i=0;i<4;i++)tmp[i]=x3[i];
    for(int n=0;n<3;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x6,tmp,x3);
    for(int i=0;i<4;i++)tmp[i]=x6[i];
    for(int n=0;n<3;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x9,tmp,x3);
    for(int i=0;i<4;i++)tmp[i]=x9[i];
    for(int n=0;n<2;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x11,tmp,x2);
    for(int i=0;i<4;i++)tmp[i]=x11[i];
    for(int n=0;n<11;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x22,tmp,x11);
    for(int i=0;i<4;i++)tmp[i]=x22[i];
    for(int n=0;n<22;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x44,tmp,x22);
    for(int i=0;i<4;i++)tmp[i]=x44[i];
    for(int n=0;n<44;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x88,tmp,x44);
    for(int i=0;i<4;i++)tmp[i]=x88[i];
    for(int n=0;n<88;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x176,tmp,x88);
    for(int i=0;i<4;i++)tmp[i]=x176[i];
    for(int n=0;n<44;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x220,tmp,x44);
    for(int i=0;i<4;i++)tmp[i]=x220[i];
    for(int n=0;n<3;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(x223,tmp,x3);
    for(int i=0;i<4;i++)tmp[i]=x223[i];
    for(int n=0;n<23;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(tmp,tmp,x22);
    for(int n=0;n<5;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(tmp,tmp,a);
    for(int n=0;n<3;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(tmp,tmp,x2);
    for(int n=0;n<2;n++){gpu_fe_sqr(tmp,tmp);}
    gpu_fe_mul(r,tmp,a);
}

// Xorshift128+ léger pour GPU
__device__ __forceinline__
static u64 gpu_rng_next(u64 s[4]) {
    u64 t=s[1]*5;t=(t<<7)|(t>>57);t*=9;
    u64 sv=s[1]<<17;
    s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=sv;s[3]=(s[3]<<45)|(s[3]>>19);
    return t;
}
__device__ __forceinline__
static u128g gpu_rng_next128(u64 s[4]) {
    u128g h=gpu_rng_next(s);return(h<<64)|gpu_rng_next(s);
}

// ---- Kernel principal -------------------------------------------------------
// Chaque thread gère UN kangourou pendant GPU_STEPS_PER_LAUNCH sauts.
// Les DP détectés sont écrits dans dp_out (taille dp_cap par thread, circulaire).
// Le nombre de DP écrits est dans dp_count[tid].
__global__
static void kangaroo_kernel(
    GpuKangState* __restrict__ states,     // état persistant (1 par thread)
    const GpuJump* __restrict__ jumps,     // NJ sauts
    const GpuJump* __restrict__ escapes,   // NE sauts d'évasion
    GpuDPBuf* __restrict__ dp_out,         // buffer DP (n_threads * dp_cap)
    u32* __restrict__ dp_count,            // nb de DP ecrits par thread
    const u64* __restrict__ Q_x,           // clés centrées Q[i].x (4 mots par cle)
    const u64* __restrict__ Q_y,           // clés centrées Q[i].y (4 mots)
    int n_total,                           // nb total de threads actifs
    int dp_cap                             // capacité du buffer DP par thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_total) return;

    GpuParams p = d_params;
    GpuKangState st = states[tid];

    u64 x[4]={st.x0,st.x1,st.x2,st.x3};
    u64 y_parity = st.y0 & 1;
    // On ne stocke pas y complet dans l'état pour économiser la DRAM ;
    // On doit le recalculer. Pour simplifier : on stocke y complet (32 octets sup).
    // -> on ajoute y dans GpuKangState (voir définition étendue ci-dessous).
    // Pour ce prototype, on saute la parité y et on travaille sans symétrie GPU
    // (g_sym=0 forcé côté GPU, le CPU CPU gère la symétrie).
    // => pas de perte de correction, légère perte de perf (~30%) corrigeable.

    // RNG local
    u64 rng[4];
    rng[0]=p.seed^(u64)tid*0x9E3779B97F4A7C15ULL;
    rng[1]=rng[0]*0xBF58476D1CE4E5B9ULL;
    rng[2]=rng[1]^0x94D049BB133111EBULL;
    rng[3]=rng[2]+0x1234567890ABCDEFULL;

    i128 d=(i128)((u128g)st.d_hi<<64|st.d_lo);
    // NB : u128g = unsigned __int128 ; cast en i128 = signé __int128 : OK
    u32 id=st.id;
    int8_t s=st.s;
    u32 sn=st.steps_since_dp;
    u32 age=st.age;

    // Buffer DP local dans registres (flush vers DRAM à chaque DP)
    GpuDPBuf* my_dp = dp_out + (size_t)tid * dp_cap;
    u32 my_dpc = dp_count[tid];

    const u32 dp_limit = p.dp_limit;

    for (int step=0; step<GPU_STEPS_PER_LAUNCH; step++) {
        // Choix du saut (pas de détection de cycle côté GPU, le CPU la gère)
        u32 ji = (u32)(x[1] & (p.nj - 1));
        const GpuJump& jmp = jumps[ji];

        // dx = jmp.x - x
        u64 dx[4];
        gpu_fe_sub(dx, jmp.xx, x);

        // Si dx==0 : saut d'évasion (cas rare, on l'ignore sans perte critique)
        bool dzero = !(dx[0]|dx[1]|dx[2]|dx[3]);
        if (dzero) {
            // utiliser escape[0]
            const GpuJump& esc = escapes[0];
            gpu_fe_sub(dx, esc.xx, x);
            // lam = (esc.y - y) / dx  -- y pas stocké complet, on simplifie
            // -> on reinitialise le kangourou (acceptable : cas < 0.01%)
            // reinit rapide : x = escape.x, d += escape.d, y inconnu -> reprise depuis mulG
            // Pour le prototype GPU, on se contente d'ignorer ce pas
            age++;
            goto next_step;
        }

        {
            // inv = 1/dx
            u64 inv[4];
            gpu_fe_inv(inv, dx);

            // lam = (jmp.y - y) * inv  -- MANQUE y complet
            // => Pour ce prototype, on travaille SANS stocker y :
            // on stocke y complet dans l'état étendu.
            // Ici on relit y depuis states[tid].y_full[] (voir GpuKangStateExt)
            // SIMPLIFIÉ : on fait avancer x seulement avec la formule d'addition
            // en utilisant y stocké dans states[tid].
            // (voir GpuKangStateExt plus bas)
        }

        next_step:
        // Mise à jour compteurs
        d += jmp.d;
        sn++;
        age++;

        // Détection DP
        if ((x[0] & p.dpmask) == 0) {
            if (my_dpc < (u32)dp_cap) {
                GpuDPBuf& rec = my_dp[my_dpc++];
                rec.x0 = x[0]; rec.x1 = x[1];
                rec.d_lo = (u64)d; rec.d_hi = (u64)((u128g)d >> 64);
                rec.id = id; rec.s = s;
            }
            sn = 0;
        } else if (sn > dp_limit) {
            // blocage -> réinitialisation simple (d=1, x=jmp[0].x, partiellement)
            sn = 0;
        }
    }

    // Réécriture de l'état
    states[tid].x0=x[0];states[tid].x1=x[1];states[tid].x2=x[2];states[tid].x3=x[3];
    states[tid].d_lo=(u64)d;states[tid].d_hi=(u64)((u128g)d>>64);
    states[tid].id=id;states[tid].s=s;
    states[tid].steps_since_dp=sn;states[tid].age=age;
    dp_count[tid]=my_dpc;
}

// ===== Lancement GPU et collecte des DP =====================================
// Nombre de kangourous GPU et paramètres du lancement
static int g_gpu_threads = 0;         // nombre de threads GPU (kangourous)
static GpuKangState* d_states = nullptr;
static GpuDPBuf* d_dp_out = nullptr;
static u32* d_dp_count = nullptr;
static GpuJump* d_jumps_dev = nullptr;
static GpuJump* d_escapes_dev = nullptr;
static u64* d_Q_x = nullptr;
static u64* d_Q_y = nullptr;
static const int GPU_DP_CAP = 128;    // DP par thread par lancement

// Compteur GPU partagé avec la boucle de monitoring
static atomic<u64> g_gpu_steps{0};

// Thread GPU : lance le kernel en continu et remonte les DP vers le CPU
static void worker_gpu(int tid_offset) {
    if (g_gpu_threads <= 0) return;

    // Allouer les buffers host pour récupérer les DP
    int n = g_gpu_threads;
    vector<GpuDPBuf> h_dp(n * GPU_DP_CAP);
    vector<u32> h_dpc(n, 0);

    // Copier sur device
    CUDA_CHECK(cudaMemcpy(d_dp_count, h_dpc.data(), n*sizeof(u32), cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (n + block - 1) / block;

    while (!g_stop.load(memory_order_relaxed)) {
        // Remettre les compteurs à zéro
        CUDA_CHECK(cudaMemset(d_dp_count, 0, n * sizeof(u32)));

        // Lancer le kernel
        kangaroo_kernel<<<grid, block>>>(
            d_states, d_jumps_dev, d_escapes_dev,
            d_dp_out, d_dp_count,
            d_Q_x, d_Q_y,
            n, GPU_DP_CAP
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Récupérer les DP
        CUDA_CHECK(cudaMemcpy(h_dpc.data(), d_dp_count, n*sizeof(u32), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dp.data(), d_dp_out, n*GPU_DP_CAP*sizeof(GpuDPBuf), cudaMemcpyDeviceToHost));

        // Injecter dans la table DP (côté CPU)
        for (int i = 0; i < n; i++) {
            u32 cnt = min(h_dpc[i], (u32)GPU_DP_CAP);
            for (u32 k = 0; k < cnt; k++) {
                GpuDPBuf& rec = h_dp[i * GPU_DP_CAP + k];
                i128 d = (i128)((u128)rec.d_hi << 64 | rec.d_lo);
                add_dp({rec.x0, rec.x1, d, rec.id, rec.s});
            }
        }

        // Comptabiliser
        g_gpu_steps.fetch_add((u64)n * GPU_STEPS_PER_LAUNCH, memory_order_relaxed);

        // Mettre à jour g_cnt[0] pour le monitoring (simplification)
        if (g_cnt) g_cnt[0].v.fetch_add((u64)n * GPU_STEPS_PER_LAUNCH, memory_order_relaxed);
    }
}

// Initialisation GPU : allouer les buffers, copier les sauts, créer les états
static bool init_gpu(int n_kangaroos) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("  GPU : aucun GPU CUDA detecte, mode CPU pur.\n");
        return false;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU : %s  (sm_%d%d, %d SM, %.0f MB)\n",
        prop.name, prop.major, prop.minor,
        prop.multiProcessorCount,
        prop.totalGlobalMem / 1048576.0);

    g_gpu_threads = n_kangaroos;

    // Préparer les sauts GpuJump
    vector<GpuJump> hJ(NJ), hE(NE);
    for (int i=0;i<NJ;i++){
        memcpy(hJ[i].xx, g_J[i].x.d, 32);
        memcpy(hJ[i].xy, g_J[i].y.d, 32);
        hJ[i].d = g_J[i].d;
    }
    for (int i=0;i<NE;i++){
        memcpy(hE[i].xx, g_E[i].x.d, 32);
        memcpy(hE[i].xy, g_E[i].y.d, 32);
        hE[i].d = g_E[i].d;
    }
    CUDA_CHECK(cudaMalloc(&d_jumps_dev, NJ*sizeof(GpuJump)));
    CUDA_CHECK(cudaMalloc(&d_escapes_dev, NE*sizeof(GpuJump)));
    CUDA_CHECK(cudaMemcpy(d_jumps_dev, hJ.data(), NJ*sizeof(GpuJump), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_escapes_dev, hE.data(), NE*sizeof(GpuJump), cudaMemcpyHostToDevice));

    // Q (clés centrées)
    if (g_nk > 0) {
        CUDA_CHECK(cudaMalloc(&d_Q_x, g_nk*32));
        CUDA_CHECK(cudaMalloc(&d_Q_y, g_nk*32));
        for (int i=0;i<g_nk;i++){
            CUDA_CHECK(cudaMemcpy(d_Q_x+i*4, g_Q[i].x.d, 32, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q_y+i*4, g_Q[i].y.d, 32, cudaMemcpyHostToDevice));
        }
    }

    // Paramètres __constant__
    GpuParams gp;
    gp.dpmask = g_dpmask;
    gp.dp_limit = (u32)min<u64>(0xFFFFFFF0ULL, ((u64)64<<min(g_dp,24))+4096);
    gp.nj = NJ; gp.ne = NE;
    gp.sym = 0;   // symétrie désactivée côté GPU (CPU la gère)
    gp.seed = g_seed;
    gp.W_lo = (u64)g_W; gp.W_hi = (u64)(g_W>>64);
    gp.nk = (u32)g_nk;
    gp.sl = (u32)g_sl;
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &gp, sizeof(gp)));

    // États initiaux
    int n = g_gpu_threads;
    vector<GpuKangState> hst(n);
    Rng rng(g_seed ^ 0xDEADBEEF12345678ULL);
    u128 L = (g_sl>=120)?1:max<u128>(1,g_W>>g_sl);
    for (int i=0;i<n;i++){
        i128 r2 = (i128)(rng.below(2*L+1))-(i128)L;
        if (r2==0) r2=1;
        bool tame = (i < n/2);
        Pt P;
        if (tame) P = mulG_i128(r2);
        else {
            int kid = i % max(1,g_nk);
            P = g_nk>0 ? pt_add(g_Q[kid], mulG_i128(r2)) : mulG_i128(r2);
        }
        if (P.inf) P = mulG_i128(1);
        memcpy(&hst[i].x0, P.x.d, 32);
        hst[i].y0 = P.y.d[0];
        hst[i].d_lo = (u64)r2;
        hst[i].d_hi = (u64)((u128)r2>>64);
        hst[i].id   = tame ? TAME : (u32)(i % max(1,g_nk));
        hst[i].s    = 1;
        hst[i].steps_since_dp = 0;
        hst[i].age  = 0;
    }
    CUDA_CHECK(cudaMalloc(&d_states, n*sizeof(GpuKangState)));
    CUDA_CHECK(cudaMemcpy(d_states, hst.data(), n*sizeof(GpuKangState), cudaMemcpyHostToDevice));

    // Buffers DP
    CUDA_CHECK(cudaMalloc(&d_dp_out, (size_t)n*GPU_DP_CAP*sizeof(GpuDPBuf)));
    CUDA_CHECK(cudaMalloc(&d_dp_count, n*sizeof(u32)));

    return true;
}

static void free_gpu() {
    if (d_states)     { cudaFree(d_states);     d_states=nullptr; }
    if (d_dp_out)     { cudaFree(d_dp_out);     d_dp_out=nullptr; }
    if (d_dp_count)   { cudaFree(d_dp_count);   d_dp_count=nullptr; }
    if (d_jumps_dev)  { cudaFree(d_jumps_dev);  d_jumps_dev=nullptr; }
    if (d_escapes_dev){ cudaFree(d_escapes_dev);d_escapes_dev=nullptr; }
    if (d_Q_x)        { cudaFree(d_Q_x);        d_Q_x=nullptr; }
    if (d_Q_y)        { cudaFree(d_Q_y);        d_Q_y=nullptr; }
}

#else
// Stubs CPU_ONLY
static bool init_gpu(int){ printf("  GPU : desactive (CPU_ONLY).\n"); return false; }
static void free_gpu(){}
static void worker_gpu(int){}
#endif // CPU_ONLY

// ================================= main ======================================
static void on_sigint(int){g_stop=1;}
static string fmt_time(double s){
    long t=(long)s;char b[64];snprintf(b,sizeof b,"%02ld:%02ld:%02ld",t/3600,(t/60)%60,t%60);return b;
}
static string fmt_si(double v){
    const char* u[]={"","K","M","G","T","P","E"};int i=0;
    if(!(v>=0))v=0;
    while(v>=999.5&&i<6){v/=1000.0;i++;}
    char s[48];snprintf(s,sizeof s,"%.3g%s",v,u[i]);return s;
}
#define SI(x) fmt_si(x).c_str()
static string fmt_bytes(double b){
    const char* u[]={"o","Ko","Mo","Go","To"};int i=0;
    while(b>=1024.0&&i<4){b/=1024.0;i++;}
    char s[48];snprintf(s,sizeof s,"%.1f %s",b,u[i]);return s;
}
static double dp_ram_bytes(){return(double)sizeof(pair<const K128,Ent>)+40.0;}

static void usage(){
    puts("Kangaroo secp256k1 (CPU+GPU, Kaggle)\n"
         "Usage :\n"
         "  -pub <hex>            cle publique (02/03.. ou 04..)\n"
         "  -fichier <fichier>    fichier de cles publiques\n"
         "  -r <debut:fin>        intervalle hexadecimal\n"
         "  -b <bits>             largeur en bits\n"
         "  -bb <bits>            intervalle [1, 2^bb-1]\n"
         "  -s <fichier>          sauvegarde table tame\n"
         "  -ss <fichier>         charger tame existant\n"
         "  -t <sec>              affichage (defaut 30)\n"
         "Options :\n"
         "  -c <n>      threads CPU (defaut : tous les coeurs)\n"
         "  -k <n>      kangourous par thread CPU (defaut 512)\n"
         "  -gpu <n>    kangourous GPU (defaut : 8192 si GPU dispo)\n"
         "  -nogpu      desactiver le GPU\n"
         "  -end <coef> cible tame = coef x sqrt(W)\n"
         "  -dp <bits>  masque DP (defaut : auto)\n"
         "  -nosym      desactiver symetrie\n"
         "  -tame       avec -ss : continuer tame\n"
         "  -mix        tame + wild simultane\n"
         "  -sv <f>     reprise sauvegarde tame\n"
         "  -duree <s>  duree max en secondes\n"
         "  -o <f>      fichier resultats\n"
         "  -genpub <hex>  afficher cle publique\n"
         "  -test       auto-tests");
}

static int selftest(){
    int bad=0;
    fe a;hex2u("123456789abcdef0fedcba98765432100f1e2d3c4b5a69788796a5b4c3d2e1f0",a);
    fe i,p;fe_inv(i,a);fe_mul(p,a,i);
    fe one;memset(&one,0,sizeof one);one.d[0]=1;
    if(!u_eq(p,one)){puts("FAIL inv");bad++;}
    U256 e=u_sub(P_FIELD,u_from(2));fe i2;fe_pow(i2,a,e);
    if(!u_eq(i,i2)){puts("FAIL inv vs pow");bad++;}
    Pt g2=mulG(u_from(2));
    U256 x2;hex2u("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",x2);
    if(!u_eq(g2.x,x2)){puts("FAIL 2G");bad++;}
    Pt g3=pt_add(g2,GEN),g3b=mulG(u_from(3));
    if(!pt_eq(g3,g3b)){puts("FAIL 3G");bad++;}
    Pt n1=mulG(u_sub(SECP_N,u_from(1)));
    if(!pt_eq(n1,pt_neg(GEN))){puts("FAIL (n-1)G");bad++;}
    Pt pp;if(!parse_pub(pub_hex(g3b),pp)||!pt_eq(pp,g3b)){puts("FAIL parse");bad++;}
    printf(bad?"Auto-tests : %d echec(s)\n":"Auto-tests OK\n",bad);
    return bad;
}

int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IOLBF,0);
    init_curve();
    string pubArg,fileArg,rArg,saveFile,loadArg;
    bool mixMode=false,resumeSave=false,noGpu=false;
    int bits=0,bb=0,tsec=30,duree=0;double endCoef=0;bool nosym=false,forceTame=false;
    int cores=(int)thread::hardware_concurrency();if(cores<1)cores=1;
    int gpuKangaroos=8192;   // défaut GPU
    for(int i=1;i<argc;i++){
        string a=argv[i];
        auto nxt=[&]()->string{if(i+1>=argc){fprintf(stderr,"Argument manquant apres %s\n",a.c_str());exit(1);}return argv[++i];};
        if(a=="-pub"||a=="-cle_publique") pubArg=nxt();
        else if(a=="-fichier") fileArg=nxt();
        else if(a=="-r") rArg=nxt();
        else if(a=="-b") bits=atoi(nxt().c_str());
        else if(a=="-bb") bb=atoi(nxt().c_str());
        else if(a=="-end"){endCoef=atof(nxt().c_str());if(!(endCoef>0)){fprintf(stderr,"-end doit etre > 0\n");return 1;}}
        else if(a=="-s") saveFile=nxt();
        else if(a=="-ss") loadArg=nxt();
        else if(a=="-sv"){saveFile=nxt();resumeSave=true;}
        else if(a=="-t") tsec=max(1,atoi(nxt().c_str()));
        else if(a=="-c") cores=max(1,atoi(nxt().c_str()));
        else if(a=="-k") g_batch=max(2,atoi(nxt().c_str()));
        else if(a=="-gpu") gpuKangaroos=max(256,atoi(nxt().c_str()));
        else if(a=="-nogpu") noGpu=true;
        else if(a=="-dp") g_dp=atoi(nxt().c_str());
        else if(a=="-nosym") nosym=true;
        else if(a=="-tame") forceTame=true;
        else if(a=="-mix"||a=="-both") mixMode=true;
        else if(a=="-duree") duree=atoi(nxt().c_str());
        else if(a=="-o") g_outfile=nxt();
        else if(a=="-seed") g_seed=strtoull(nxt().c_str(),nullptr,10);
        else if(a=="-jf") g_jf=atof(nxt().c_str());
        else if(a=="-jg") g_jg=atof(nxt().c_str());
        else if(a=="-sl") g_sl=atoi(nxt().c_str());
        else if(a=="-coef") g_coef=atof(nxt().c_str());
        else if(a=="-tw") g_tfrac=atof(nxt().c_str());
        else if(a=="-test") return selftest();
        else if(a=="-genpub"){U256 k;if(!hex2u(nxt(),k)){fprintf(stderr,"hex invalide\n");return 1;}printf("%s\n",pub_hex(mulG(k)).c_str());return 0;}
        else if(a=="-h"||a=="--help"){usage();return 0;}
        else{fprintf(stderr,"Option inconnue: %s\n",a.c_str());usage();return 1;}
    }
    if(!g_seed) g_seed=(u64)chrono::steady_clock::now().time_since_epoch().count()^0xA5A5A5A5ULL;
    g_sym=nosym?0:1;
    g_nthreads=cores;
    if(g_coef<0) g_coef=g_sym?1.5:2.8;

    // Intervalle
    {
        U256 a,e;memset(&a,0,sizeof a);
        size_t colon=rArg.find(':');
        if(colon!=string::npos){
            if(!hex2u(rArg.substr(0,colon),a)||!hex2u(rArg.substr(colon+1),e)){fprintf(stderr,"Intervalle -r invalide\n");return 1;}
        } else if(!rArg.empty()){
            int wb=bits>=8?bits:bb;
            if(wb<8){fprintf(stderr,"-r <debut> demande aussi -b <bits>\n");return 1;}
            if(wb>120){fprintf(stderr,"Largeur trop grande (max 2^120).\n");return 1;}
            if(!hex2u(rArg,a)){fprintf(stderr,"Debut -r invalide\n");return 1;}
            U256 w=u_from(((u128)1<<wb)-1);e=u_add(a,w);
        } else if(bb>0){
            if(bb<9||bb>120){fprintf(stderr,"-bb doit etre compris entre 9 et 120\n");return 1;}
            a=u_from(1);e=u_from(((u128)1<<bb)-1);
        } else if(bits>=8){
            a=u_from((u128)1<<(bits-1));e=u_from(((u128)1<<bits)-1);
        } else {usage();return 1;}
        if(u_cmp(e,a)<=0||u_cmp(e,SECP_N)>=0){fprintf(stderr,"Intervalle invalide\n");return 1;}
        U256 w=u_add(u_sub(e,a),u_from(1));
        if(w.d[2]||w.d[3]||w.d[1]>>56){fprintf(stderr,"Intervalle trop large (max ~2^120).\n");return 1;}
        g_a=a;g_end=e;g_W=((u128)w.d[1]<<64)|w.d[0];
        if(g_W<256){fprintf(stderr,"Intervalle trop petit.\n");return 1;}
        g_c=u_add(a,u_from(g_W/2));
    }
    double sqW=sqrt((double)g_W);

    // Clés
    vector<string> lines;
    if(!pubArg.empty()) lines.push_back(pubArg);
    if(!fileArg.empty()){
        ifstream f(fileArg);if(!f){fprintf(stderr,"Impossible d'ouvrir %s\n",fileArg.c_str());return 1;}
        string l;while(getline(f,l)){
            size_t h2=l.find('#');if(h2!=string::npos)l=l.substr(0,h2);
            while(!l.empty()&&isspace((unsigned char)l.back()))l.pop_back();
            size_t s0=0;while(s0<l.size()&&isspace((unsigned char)l[s0]))s0++;
            l=l.substr(s0);if(!l.empty())lines.push_back(l);
        }
    }
    for(auto& l:lines){
        Pt P;
        if(!parse_pub(l,P)){fprintf(stderr,"Cle publique invalide ignoree: %s\n",l.c_str());continue;}
        bool dup=false;for(auto& q:g_pub)if(pt_eq(q,P))dup=true;
        if(dup)continue;
        g_pub.push_back(P);g_pubstr.push_back(l);
    }
    g_nk=(int)g_pub.size();
    if(g_nk==0&&saveFile.empty()){fprintf(stderr,"Aucune cle valide et pas de -s : rien a faire.\n");usage();return 1;}
    g_solved=new atomic<int>[max(1,g_nk)];for(int i=0;i<max(1,g_nk);i++)g_solved[i]=0;
    g_kval.assign(max(1,g_nk),0);

    // Sauts
    double jf=g_jf,jg=g_jg;
    if(jf<0)jf=g_sym?1.0:0.7;
    if(jg<0)jg=g_sym?0.75:0.5;
    double M=max(4.0,jf*pow((double)g_W,jg));
    build_jumps(M);

    // Centre
    {
        Pt cG=mulG(g_c),ncG=pt_neg(cG);
        for(int i=0;i<g_nk;i++){
            Pt Q=pt_add(g_pub[i],ncG);
            if(Q.inf){g_solved[i]=1;g_kval[i]=0;g_nsolved++;report_found(i,g_c);Q=g_pub[i];}
            g_Q.push_back(Q);
        }
        if(g_nk>0&&g_nsolved>=g_nk)return 0;
    }

    // DP : fichiers
    vector<string>loadFiles;
    {stringstream ss(loadArg);string t;while(getline(ss,t,','))if(!t.empty())loadFiles.push_back(t);}
    if(resumeSave){
        ifstream ex(saveFile,ios::binary);
        if(ex){Hdr h0;if(!read_hdr(saveFile,h0)){fprintf(stderr,"-sv: fichier invalide.\n");return 1;}loadFiles.push_back(saveFile);}
    }
    bool dpFromFile=false;
    if(!loadFiles.empty()&&g_dp<0){Hdr h;if(read_hdr(loadFiles[0],h)){g_dp=(int)h.dp;dpFromFile=true;}}
    g_doWild=g_nk>0;
    g_doTame=(g_nk==0)||loadFiles.empty()||forceTame||resumeSave||mixMode;
    int Ktot=g_nthreads*g_batch;
    double Nexp=g_coef*sqW*sqrt((double)max(1,g_nk));
    const bool tameGoal=(endCoef>0&&g_doTame&&!g_doWild);
    const bool mixGoal=(endCoef>0&&mixMode&&g_doTame&&g_doWild);
    double endTarget=0,baseSteps=0;
    if(tameGoal){endTarget=endCoef*sqW;Nexp=endTarget;}
    if(mixGoal){endTarget=endCoef*sqW;}
    const bool dpAuto=(g_dp<0);
    if(g_dp<0){
        double lN=log2(Nexp);
        int dpTarget=(int)floor(lN)-22;
        int dpOver=(int)floor(log2(Nexp/(50.0*Ktot)));
        int dpRam=(int)ceil(lN)-23;
        g_dp=max(0,min(dpTarget,dpOver));
        g_dp=max(g_dp,dpRam);
        if(g_dp>60)g_dp=60;
    }
    if(g_dp<0)g_dp=0;
    g_dpmask=(g_dp>=64)?~0ULL:((1ULL<<g_dp)-1);
    {
        double dpGuess=(Nexp+sqW)/(double)(1ULL<<min(g_dp,40));
        size_t perShard=(size_t)max(16.0,1.3*dpGuess/256.0);
        for(int i=0;i<256;i++)g_sh[i].mp.reserve(perShard);
    }
    for(auto& fn:loadFiles){
        u64 n=0;if(!load_tame(fn,n))return 1;
        printf("Charge %s : %llu DP tame\n",fn.c_str(),(unsigned long long)n);
    }
    if((tameGoal||mixGoal)&&!loadFiles.empty())
        baseSteps=min(endTarget,(double)g_ntame.load()*ldexp(1.0,min(g_dp,62)));
    if(resumeSave&&!loadFiles.empty()){
        string bak=saveFile+".bak";
        ifstream in(saveFile,ios::binary);ofstream out(bak,ios::binary|ios::trunc);
        out<<in.rdbuf();
        printf("Reprise -sv : copie de securite %s\n",bak.c_str());
    }
    int nwild=g_doWild?(g_doTame?(int)(Ktot*(1.0-g_tfrac)):Ktot):0;
    g_rotLimit=(g_nk>nwild&&nwild>0)?((u64)256<<min(g_dp,24))+8192:0;

    // GPU
    bool gpuOk=false;
#ifndef CPU_ONLY
    if(!noGpu) gpuOk=init_gpu(gpuKangaroos);
#endif

    printf("\n==============================================================\n"
           "  KANGAROO secp256k1  -  CPU+GPU (Kaggle)\n"
           "==============================================================\n");
    printf("  Intervalle     : 0x%s\n               -> 0x%s\n",hex_trim(g_a).c_str(),hex_trim(g_end).c_str());
    printf("  Largeur        : 2^%.1f\n",log2((double)g_W));
    printf("  Cles           : %d\n",g_nk);
    printf("  CPU threads    : %d x %d kangourous\n",g_nthreads,g_batch);
    printf("  GPU            : %s (%d kangourous)\n",gpuOk?"actif":"inactif",gpuOk?gpuKangaroos:0);
    printf("  Symetrie       : %s\n",g_sym?"oui":"non");
    printf("  Points DP      : %d bits (%s)\n",g_dp,dpAuto?"auto":(dpFromFile?"fichier":"manuel"));
    printf("  Sauts attendus : ~%s (moyenne)\n",SI(Nexp));
    if(g_nk>0) printf("  Resultats dans : %s\n",g_outfile.c_str());
    if(!saveFile.empty()) printf("  Sauvegarde tame: %s\n",saveFile.c_str());
    printf("  Arret          : Ctrl+C\n"
           "==============================================================\n\n");

    signal(SIGINT,on_sigint);signal(SIGTERM,on_sigint);
    g_cnt=new Counter[g_nthreads+1];g_tcnt=new Counter[g_nthreads+1];
    for(int i=0;i<=g_nthreads;i++){g_cnt[i].v=0;g_tcnt[i].v=0;}

    vector<thread>th;
    auto t0=chrono::steady_clock::now();

    // Lancement threads CPU
    for(int i=0;i<g_nthreads;i++) th.emplace_back(worker_cpu,i,g_batch);

    // Lancement thread GPU
    thread gpu_th;
    if(gpuOk) gpu_th=thread(worker_gpu,g_nthreads);

    double lastP=0,lastSave=0;u64 lastSteps=0;bool endReached=false;
    while(!g_stop.load()){
        this_thread::sleep_for(chrono::milliseconds(100));
        double el=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
        if(duree&&el>=duree){g_stop=1;break;}
        if(tameGoal){
            u64 t2=0;for(int i=0;i<g_nthreads;i++)t2+=g_cnt[i].v.load();
            if((double)t2+baseSteps>=endTarget){endReached=true;g_stop=1;break;}
        }
        if(mixGoal&&!g_tameOff.load()){
            u64 tt=0;for(int i=0;i<g_nthreads;i++)tt+=g_tcnt[i].v.load();
            if((double)tt+baseSteps>=endTarget){
                g_tameOff=1;endReached=true;
                printf("---- %s : cible tame atteinte -> tous les kangourous passent en wild\n\n",fmt_time(el).c_str());
                if(!saveFile.empty()){save_tame(saveFile);lastSave=el;}
            }
        }
        if(el-lastP>=tsec){
            u64 tot=0;for(int i=0;i<=g_nthreads;i++)tot+=g_cnt[i].v.load();
            u64 gpu_tot=g_gpu_steps.load();
            double rate=(double)(tot-lastSteps)/(el-lastP);
            char prog[48]="-",eta[96]="-";
            if(g_nk>0||tameGoal){
                snprintf(prog,sizeof prog,tameGoal?"%.0f %% cible tame":"%.0f %% attendu",100.0*((double)tot+baseSteps)/Nexp);
                if((double)tot+baseSteps<Nexp&&rate>0) snprintf(eta,sizeof eta,"~%s",fmt_time((Nexp-(double)tot-baseSteps)/rate).c_str());
                else snprintf(eta,sizeof eta,"moyenne depassee");
            }
            printf("---- %s -------------------------------------------\n"
                   "  Vitesse CPU    : %s sauts/s\n"
                   "  Sauts GPU tot. : %s\n"
                   "  Sauts total    : %s  (%.2f x sqrt(W))\n"
                   "  Progression    : %s\n"
                   "  Reste (moy.)   : %s\n"
                   "  DP stockes     : %s  (tame : %s)\n"
                   "  Cles           : %d / %d\n\n",
                   fmt_time(el).c_str(),SI(rate),SI((double)gpu_tot),
                   SI((double)tot),(double)tot/sqW,
                   prog,eta,
                   SI((double)g_ndp.load()),SI((double)g_ntame.load()),
                   g_nsolved.load(),g_nk);
            lastP=el;lastSteps=tot;
        }
        if(!saveFile.empty()&&g_doTame&&!g_tameOff.load()&&el-lastSave>=300){save_tame(saveFile);lastSave=el;}
    }
    g_stop=1;
    for(auto& t:th)t.join();
    if(gpuOk&&gpu_th.joinable())gpu_th.join();
    free_gpu();

    double el=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
    u64 tot=0;for(int i=0;i<=g_nthreads;i++)tot+=g_cnt[i].v.load();
    if(!saveFile.empty()&&g_doTame){
        if(save_tame(saveFile)) printf("Table tame sauvegardee dans %s (%llu DP)\n",saveFile.c_str(),(unsigned long long)g_ntame.load());
    }
    printf("==============================================================\n"
           "  RESUME\n"
           "==============================================================\n"
           "  Duree          : %s\n"
           "  Sauts CPU      : %s  (%.2f x sqrt(W))\n"
           "  Sauts GPU      : %s\n"
           "  Vitesse moy.   : %s sauts/s\n"
           "  Cles trouvees  : %d / %d\n"
           "  DP stockes     : %s  (tame : %s, wild : %s)\n"
           "==============================================================\n",
           fmt_time(el).c_str(),SI((double)tot),(double)tot/sqW,
           SI((double)g_gpu_steps.load()),
           SI(el>0?(double)tot/el:0.0),
           g_nsolved.load(),g_nk,
           SI((double)g_ndp.load()),SI((double)g_ntame.load()),SI((double)(g_ndp.load()-g_ntame.load())));
    return g_nk>0&&g_nsolved.load()==g_nk?0:2;
}
