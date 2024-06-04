// Author: Zhen Zheng
// To be used in the future as a tool to generating the FP6 matrix from the FP16 matrix.

#include<iostream>

/*
 * Function to pack 4 fake quantized FP16 value into continuously stored 4 FP6 values.
 */
void cast_fp16_fp6(uint16_t* FP16x4, uint8_t* FP6x4)
{
    // Constants for FP6
    constexpr int exponent_nbits_fp6 = 3;
    constexpr int mantissa_nbits_fp6 = 2;
    constexpr int exp_bias_fp6 = (1 << (exponent_nbits_fp6 - 1)) - 1;
    // Constants for FP16
    constexpr int exponent_nbits_fp16 = 5;
    constexpr int mantissa_nbits_fp16 = 10;
    constexpr int exp_bias_fp16 = (1 << (exponent_nbits_fp16 - 1)) - 1;

    int fp6_temp[4];

    float absmin_nonzero_fp6 = 0.0625;
    // Note that we regard the exponent of '111' as a regular value rather than NaN or inf. This is
    // the same with that in qtorch.
    float absmax_fp6 = 28;

    for (int i = 0; i < 4; ++i) {
        uint16_t source = FP16x4[i];
        float fp6_value_abs = std::abs(__half2float(*((half*)(&source))));
        if ((fp6_value_abs != 0 && fp6_value_abs < absmin_nonzero_fp6) ||
            fp6_value_abs > absmax_fp6) {
            // TODO(zhen): a better way may be rounding it to the nearest FP6 value.
            throw std::invalid_argument("Input value out of range for FP6.");
        }

        // It is not safe to do shift operation on uint16_t. So we promote it to int.
        int source_promote = int(source);

        int sign_bit = (source_promote >> 15);
        // Extracting exponent represented in FP16. The sign mask 0x7FFF is '0111 1111 1111 1111'
        int exp_bit = (source_promote & 0x7FFF) >> mantissa_nbits_fp16;
        // Extracting mantissa represented in FP16
        int mant_bit = source_promote & ((1 << mantissa_nbits_fp16) - 1);

        int new_exp_bit;
        int new_mant_bit;

        if (exp_bit == 0) {
            // Subnormal FP16 number. Too small for FP6.
            new_exp_bit = 0;
            new_mant_bit = 0;
        } else {
            new_mant_bit = mant_bit >> (mantissa_nbits_fp16 - mantissa_nbits_fp6);
            new_exp_bit = exp_bit - exp_bias_fp16 + exp_bias_fp6;

            // Deal with subnormal FP6 values.
            int target_exp_val = exp_bit - exp_bias_fp16;
            int min_fp6_exp_val = -exp_bias_fp6 + 1;
            bool subnormal_fp6 = target_exp_val < min_fp6_exp_val;
            if (subnormal_fp6) {
                // TODO(zhen): add the rounding logic.
                new_exp_bit = 0;
                // The implicit 1 in the mantissa of FP16 is not present in subnormal FP6. Thus we
                // need to add it
                new_mant_bit = (new_mant_bit | (1 << mantissa_nbits_fp6)) >>
                               (min_fp6_exp_val - target_exp_val);
            }
        }

        fp6_temp[i] = (sign_bit << (exponent_nbits_fp6 + mantissa_nbits_fp6)) |
                      (new_exp_bit << mantissa_nbits_fp6) | new_mant_bit;
    }
    // Pack the values
    FP6x4[0] = fp6_temp[0] << 2 | (fp6_temp[1] >> 4);
    FP6x4[1] = (fp6_temp[1] & 0x0F) << 4 | (fp6_temp[2] >> 2);
    FP6x4[2] = (fp6_temp[2] & 0x03) << 6 | fp6_temp[3];
}

/*
 *  Function to prepack FP16 weights into continuous FP6 values.
 *
 *  Parameters:
 *     weight_16bit: input weight in FP16, size M*K
 *     weight_6bit: output weight in packed FP6, continuously stored, size M*K*6/8
 *     M, K: the shape of the weight
 */
void weight_prepacking_fp16_to_fp6(uint16_t* weight_16bit,
                                   uint8_t* weight_6bit_packed,
                                   size_t M,
                                   size_t K)
{
    // Every four 16-bit elements are packed into three 6-bit values (4*6bit == 3*8bit).
    if (K * 6 % 8 != 0) { throw std::invalid_argument("(K * 6 % 8) should be 0"); }
    size_t K_fp6_packed = K * 6 / 8;
    // #pragma omp parallel for
    for (auto m = 0; m < M; m++) {
        uint8_t* ptr_6bit = weight_6bit_packed + m * K_fp6_packed;
        uint16_t* ptr_16bit = weight_16bit + m * K;
        for (auto k = 0; k < K; k += 4) {
            cast_fp16_fp6(ptr_16bit, ptr_6bit);
            ptr_16bit += 4;
            ptr_6bit += 3;
        }
    }
}