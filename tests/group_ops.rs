use std::{clone, string, vec, iter::Skip};
use ark_std::{rand::{rngs::StdRng,RngCore},test_rng,ops::{Mul,Add,Sub,Div}, UniformRand};
use ark_ff::{Field,PrimeField,FpParameters, SquareRootField, bytes,Zero,One,Fp256, BigInteger, BigInteger256,FftField};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
    UVPolynomial,evaluations
};
use ark_ec::{group::Group,msm::{FixedBaseMSM, VariableBaseMSM}, AffineCurve, bls12::{G1Projective, G1Affine}, ProjectiveCurve};
use ark_bls12_381::{Bls12_381, Fr as F, Fq, G1Projective as G1, G2Projective as G2, fq};
use linear_algebra_modules::vector_ops::*;
use rand::Rng;
use field_matrix_utils::{self, Matrix};

#[test]   
///Checks if g+g = 2.g
pub fn doubling_check(){
  let mut rng = test_rng();
  let gen:G1 = G1::rand(& mut rng);
  let sc_test = gen+gen; 
  let doub = Group::double(&gen);
  assert_eq!(sc_test,doub); 
}


#[test]   
///checks basic associativity 
/// as: sum_i (a_i).(tau^i.g)  = (sum_i (a_i.tau^i)).g 
pub fn group_field_associativity_check(){

let mut rng = test_rng();
let mut fld =F::rand(&mut rng);
let gen:G1 = G1::rand(& mut rng);

let init_domain_size = 256;
let test_vec:Vec<F> = gen_rand_vec(256, &mut rng);
let init_domain= GeneralEvaluationDomain::<F>::new(init_domain_size).unwrap();
  // println!("Expected domain size is {:?}",init_domain);
//go to poly coeff form
let test_coeffs = GeneralEvaluationDomain::ifft(&init_domain,&test_vec);
let p_of_tau: Vec<F> = gen_pow_tau(init_domain_size,fld);

// sum_i a_i g tau^i
// this is n group M, and n-1 group add for i=0,..,n-1.
let mut srs = Vec::new();
for _i in 0..=init_domain_size-1 {
  srs.push(Group::mul(&gen,&p_of_tau[_i]));
}
let mut comm1 = G1::zero();
for _i in 0..=init_domain_size-1 {
  comm1+= Group::mul(&srs[_i],&test_coeffs[_i]);
}

// in general this is not possible to do because, the SRS is supposed to hide the secret
//scalar and no one knows it. note however it is one group M 
let scalar_mult = vec_hadamard_prod(test_coeffs,p_of_tau);
let mut comm2 = G1::zero();
let mut inter= vector_trace(scalar_mult);
comm2+= Group::mul(&gen,&inter); 
assert_eq!(comm1,comm2);
println!("Associativity check passed!");
}
