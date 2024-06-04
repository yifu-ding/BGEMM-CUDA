#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <bitset>
// #include <string>
// using namespace std;


void print_binary(unsigned char *inp, const char *array_name, int M, int K) {
    printf("%s\n", array_name);
    for (int i=1; i<=M*K/8; ++i) {
        std::bitset<8> bits(inp[i-1]);
        printf("%s ", bits.to_string().c_str());
        if (i % (M / 8) == 0) {
            printf("\n");  // M elements per row
        }
        // if (i == 2) {
        //     printf("\n");
        //     return;
        // }
    }
    printf("\n");
}


void print_binary(half *inp, const char *array_name, int M, int K) {
    /* printf("%s\n", array_name);
    for (int i=1; i<=M*K; ++i) {
        printf("%d", (int) inp[i-1]);
        if (i % 8 == 0) printf(" ");
        if (i % M == 0) {
            printf("\n");  // M elements per row
        }
        
        // if (i == 16) {
        //     printf("\n");
        //     return;
        // }
    }
    printf("\n"); */
    FILE *fp;//文件指针
    if (array_name[0] == 'A') {
        fp = fopen ("A_bin_bug.txt", "w+");
    } else {
        fp = fopen ("B_bin_bug.txt", "w+");
    }
    for (int i=1; i<=M*K; ++i) {
        fprintf(fp, "%d", (int) inp[i-1]);
        if (i % 32 == 0) fprintf(fp, " ");
        if (i % K == 0) {
            fprintf(fp, "\n");  // M elements per row
        }
    }
    fclose(fp);
}

void print_uint32(uint32_t *inp, const char *array_name, int M, int K) {
    /* printf("%s\n", array_name);
    for (int i=1; i<=M*K/32; ++i) {
        printf("%zu ", (uint32_t) inp[i-1]);
        if (i % (K/32) == 0) printf("\n");
        // if (i % M == 0) {
        //     printf("\n");  // M elements per row
        // }
    }
    printf("\n"); */
    FILE *fp;//文件指针
    if (array_name[0] == 'A') {
        fp = fopen ("A_1bit_bug.txt", "w+");
    } else {
        fp = fopen ("B_1bit_bug.txt", "w+");
    }
    for (int i=1; i<=M*K/32; ++i) {
        fprintf(fp, "%zu, ", (uint32_t) inp[i-1]);
        if (i % (K/32) == 0) fprintf(fp, "\n");
    }
    fclose(fp);
}


void print_half(half *inp, const char *array_name, int M, int N) {
    printf("%s\n", array_name);
    for (int i=1; i<=M*N; ++i) {
        printf("%d ", (int) __half2float(inp[i-1]));
        if (i % 8 == 0) printf("\n");
        // if (i % M == 0) {
        //     printf("\n");  // M elements per row
        // }
    }
    printf("\n");
}



union FloatToUint32 {
  float f;
  uint32_t u;
};

float uint32_to_float(uint32_t u) {
    FloatToUint32 converter;
    converter.u = u;
    float my_float = converter.f;
    return my_float;
}

uint32_t float_to_uint32(float f){
    FloatToUint32 converter;
    converter.f = f;
    uint32_t my_uint32_t = converter.u;
    return my_uint32_t;
}
