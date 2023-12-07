#include "PPintrin.h"
#include "math.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}
// 給定Vector以及指數，對vector內的每個element計算其對應到的指數，若>9.999，則飽和
void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float val;
  __pp_vec_int exp;
  __pp_vec_int cnt;
  __pp_vec_float result = _pp_vset_float(1.0);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float max_mask = _pp_vset_float(9.999999);
  __pp_mask maskAll, maskZero, maskNotZero, maskMult, maskSaturate;
  int remain = N%VECTOR_WIDTH;
  int cnt_loop = 0;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Initialize
    // to adjust the length of maskAll when the input vector will less than vector_width
    if (cnt_loop == N/VECTOR_WIDTH && remain != 0){
      maskAll = _pp_init_ones(remain);
    }else{
      maskAll = _pp_init_ones();
    }
    maskZero = _pp_init_ones(0);
    maskMult = _pp_init_ones(0);
    maskSaturate = _pp_init_ones(0);

    //load values and exponents to vector 
    _pp_vload_float(val, values+i, maskAll);
    _pp_vload_int(exp, exponents+i, maskAll);

    // if the exponent is 0, than set the value 1.0 to result 
    _pp_veq_int(maskZero, exp, zero, maskAll);
    //_pp_vset_float(result, 1.0, maskZero);

    // else (exponent greater than 0)
    maskNotZero = _pp_mask_not(maskZero);
    _pp_vload_float(result, values+i, maskNotZero);
    _pp_vsub_int(cnt, exp, one, maskNotZero);

    // jugde whether the exponents are greater tnan 0, if true conduct multiply
    _pp_vgt_int(maskMult, cnt, zero, maskAll);
    // count non zero element in maskMult, if mask is all false, it means no multiplicaton will be conduct
    int num_nonzero = _pp_cntbits(maskMult);
    // printf("non zero element: %d\n", num_nonzero);

    while(num_nonzero > 0){
      _pp_vmult_float(result, result, val, maskMult);
      _pp_vsub_int(cnt, cnt, one, maskMult);
      //saturation mask
      _pp_vgt_float(maskSaturate, result, max_mask, maskAll);
      _pp_vset_float(result, 9.999999, maskSaturate);
      // for stop criterion
      _pp_vgt_int(maskMult, cnt, zero, maskAll);
      num_nonzero = _pp_cntbits(maskMult);
    }

    _pp_vset_float(result, 1.0, maskZero);
    _pp_vstore_float(output+i, result, maskAll);

    cnt_loop++;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float pp_sum(__pp_vec_float array, int times){
  if (times == 0){
    //printf("%f\n", array.value[0]);
    return array.value[0];
  }else{
    __pp_vec_float sumResult=_pp_vset_float(0.0);
    _pp_hadd_float(sumResult, array);
    _pp_interleave_float(sumResult, sumResult);
    return pp_sum(sumResult, times-1);
  }
}

float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float val;
  __pp_mask maskAll;
  int times = log2(VECTOR_WIDTH);
  float sum = 0;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    _pp_vload_float(val, values+i, maskAll);
    sum = sum + pp_sum(val, times);
    // printf("sum: %f \n", sum);

  }

  return sum;
}

