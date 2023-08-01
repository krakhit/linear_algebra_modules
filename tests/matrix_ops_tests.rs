#[cfg(test)]
use ark_ff::{Field,PrimeField,FpParameters, SquareRootField, bytes,Zero,One,Fp256, BigInteger, BigInteger256,FftField};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
    UVPolynomial,evaluations
};
use ark_std::{rand::{rngs::StdRng,RngCore},test_rng,ops::{Mul,Add,Sub,Div}, UniformRand, log2};
use ark_bls12_381::Fr as F;
use linear_algebra_modules::vector_ops::*;
use std::time::{Duration,Instant};

/// this function checks the function create toeplitz
#[test]
pub fn toeplitz_check(){
  let mut rng = test_rng();
  // /131072\
  let fld = F::rand(& mut rng);
  let vec_size = 4;
  let powers:Vec<F> = gen_pow_tau(4, fld);
 
  let mut ToeplitzM = vec![];
  //assume square matrix
  ToeplitzM.push(powers.clone());
  //row counter in toeplitz M, row 0 is filled
  for i in 1..=vec_size-1 {
    let mut row=Vec::<F>::new();
  //column counter for zero
    for j in 0..i{
      row.push(F::zero());
    }
    for _k in 0..vec_size-i {
      row.push(powers[_k].clone());
    }
    ToeplitzM.push(row.clone());
    drop(row);
  }
//check matrix has toeplitz structure
for i in 0..=vec_size-2{
  for j in 0..=vec_size-2{
    assert_eq!(ToeplitzM[i][j].clone(),ToeplitzM[i+1][j+1].clone())
  }
}

// check the create random toeplitz function
let mut test_matrix:Vec<Vec<F>> = upper_triangular_toeplitz(4,fld);
for i in 0..=vec_size-2{
  for j in 0..=vec_size-2{
    //this checks the M(i,j) = M(i+1,j+1)
    assert_eq!(test_matrix[i][j].clone(),test_matrix[i+1][j+1].clone())
  }
}
// check the random toeplitz function produces the same matrix.
for i in 0..=vec_size-1{
    assert_vector_eq(ToeplitzM[i].clone(),test_matrix[i].clone());
}

}

#[test]
pub fn toeplitz_multiply(){
  let mut rng = test_rng();
  let fld = F::rand(& mut rng);
  let domain:usize = 8192;

  let test_matrix:Vec<Vec<F>> = upper_triangular_toeplitz(domain, fld);
  let test_vec:Vec<F> = gen_pow_tau(domain, fld);
   

  let start1 = Instant::now();
  let naive_result = matrix_vec_multiply(test_matrix.clone(),test_vec.clone());
  let naive_duration = start1.elapsed();

  let mut circulant_Vector:Vec<F>= Vec::<F>::new();
  let do_len = 2*domain;
  let half_len = do_len/2;

  for i in 0..=do_len-1{
    if i<= do_len/2-1 {
    circulant_Vector.push(test_matrix[i][0].clone());
    } else if i==(do_len/2){
      circulant_Vector.push(F::one());
    } else {
    circulant_Vector.push(test_matrix[i-half_len-1][half_len-1].clone());
    }
  }

  let start2 = Instant::now();
  let ext_vec = zero_pad(do_len, test_vec.clone());
  let init_domain_size = do_len.clone();
  let init_domain= GeneralEvaluationDomain::<F>::new(init_domain_size).unwrap();
  let fft1 =GeneralEvaluationDomain::fft(&init_domain,&ext_vec);
  let fft2 =GeneralEvaluationDomain::fft(&init_domain,&circulant_Vector);
  let res1 = vec_hadamard_prod(fft1, fft2);
  let fin_res= GeneralEvaluationDomain::ifft(&init_domain,&res1);
  let circulant_duration = start2.elapsed();

  let mut circulant_res = Vec::<F>::new();
  for _i in 0..=half_len-1 {
    circulant_res.push(fin_res[_i])
  }

//  assert_vector_eq(circulant_res, naive_result)
  println!("Vector size in powers of 2 {:?}", log2(domain));
  println!("Time taken for naive mult in M1 {:?} ", naive_duration);
  println!("Time taken for circulant mult in M1 {:?} ", circulant_duration);

// assert_eq!(fin_2[6],result[1]);
// assert_eq!(fin_2[5],result[2]);
// assert_eq!(fin_2[4],result[3]);
//println!("{:?}",circulant_Vector);
// println!("{:?}",test_matrix[2]);
}