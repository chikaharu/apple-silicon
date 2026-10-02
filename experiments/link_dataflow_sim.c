#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

typedef struct {
    double ops;
    double memory_words;
    double wire_hops;
    double latency_units;
} Cost;

static Cost gpu_dense(int n, int layers) {
    Cost c = {0};
    for (int l=0;l<layers;l++) {
        c.ops += 2.0*n*n;
        c.memory_words += n*n + 2.0*n;
        c.wire_hops += 2.0*n*n;
        c.latency_units += n;
    }
    return c;
}

static Cost link_lowrank(int n, int rank, int layers, int avg_hops) {
    Cost c = {0};
    for (int l=0;l<layers;l++) {
        // y = U(V^T x): two rank-n contractions
        c.ops += 4.0*n*rank;
        c.memory_words += 2.0*n*rank + 2.0*n;
        c.wire_hops += 2.0*n*rank*avg_hops;
        c.latency_units += 2.0*rank + avg_hops;
    }
    return c;
}

static void print_case(int n,int r,int layers,int hops) {
    Cost g=gpu_dense(n,layers);
    Cost l=link_lowrank(n,r,layers,hops);
    printf("n=%d rank=%d layers=%d hops=%d\n",n,r,layers,hops);
    printf("gpu_ops=%.0f link_ops=%.0f ops_reduction=%.6f\n",g.ops,l.ops,g.ops/l.ops);
    printf("gpu_mem=%.0f link_mem=%.0f mem_reduction=%.6f\n",g.memory_words,l.memory_words,g.memory_words/l.memory_words);
    printf("gpu_hops=%.0f link_hops=%.0f hop_reduction=%.6f\n",g.wire_hops,l.wire_hops,g.wire_hops/l.wire_hops);
    printf("gpu_latency=%.0f link_latency=%.0f latency_reduction=%.6f\n",g.latency_units,l.latency_units,g.latency_units/l.latency_units);
    printf("---\n");
}

int main(void){
    int ns[] = {64,128,256,512,1024};
    int rs[] = {1,2,4,8,16,32};
    int layers=32;
    int hops=2;
    for(size_t i=0;i<sizeof(ns)/sizeof(ns[0]);i++){
        for(size_t j=0;j<sizeof(rs)/sizeof(rs[0]);j++){
            if(rs[j] <= ns[i]/2) print_case(ns[i],rs[j],layers,hops);
        }
    }
    return 0;
}
