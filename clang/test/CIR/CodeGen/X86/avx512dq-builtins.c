// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-cir -o %t.cir -Wall -Werror 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

__m512i test_mm512_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm512_movm_epi64
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u8i), !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 8>), !cir.vector<!s64i x 8>
  // LLVM-LABEL: @test_mm512_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i64>
  return _mm512_movm_epi64(__A); 
}

__mmask16 test_mm512_movepi32_mask(__m512i __A) {
  // CIR-LABEL: _mm512_movepi32_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s32i x 16>, !cir.vector<!cir.int<u, 1> x 16>

  // LLVM-LABEL: @test_mm512_movepi32_mask
  // LLVM: [[CMP:%.*]] = icmp slt <16 x i32> %{{.*}}, zeroinitializer
  return _mm512_movepi32_mask(__A); 
}

__mmask8 test_mm512_movepi64_mask(__m512i __A) {
  // CIR-LABEL: @_mm512_movepi64_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s64i x 8>, !cir.vector<!cir.int<u, 1> x 8>

  // LLVM-LABEL: @test_mm512_movepi64_mask
  // LLVM: [[CMP:%.*]] = icmp slt <8 x i64> %{{.*}}, zeroinitializer
  return _mm512_movepi64_mask(__A); 
}
