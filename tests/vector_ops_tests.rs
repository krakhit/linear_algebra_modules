#[cfg(test)]
use rayon::prelude::*;
use ark_ff::{Field,PrimeField,FpParameters, SquareRootField, bytes,Zero,One,Fp256, BigInteger, BigInteger256,FftField};
    use ark_std::{rand::{rngs::StdRng,RngCore},test_rng,ops::{Mul,Add,Sub,Div}, UniformRand};
use ark_bls12_381::Fr as F;
use linear_algebra_modules::vector_ops::*;

    //use ark_poly::{EvaluationDomain, GeneralEvaluationDomain}; 

// checks the gen_rand_vec function.
#[test]   
pub fn create_random_vec(){
    let n=4;
    let mut rng = test_rng();
    let ran_vec:Vec<F> = gen_rand_vec::<F,StdRng>(n,&mut rng);
    assert_eq!(ran_vec.len(),n);
    println!("gen_random vec test passed");
}
//tests basic vector ops and assert vector eq function
#[test]   
pub fn test_vector_ops(){
    let n=4;
    let mut rng = test_rng();
    let ran_vec:Vec<F> = gen_rand_vec::<F,StdRng>(n,&mut rng);
    // generate vectors a_i, b_i
    let ran_vec1 = gen_rand_vec::<F,StdRng>(n, &mut rng);
    let ran_vec2 = gen_rand_vec::<F,StdRng>(n, &mut rng);
    let mut add_res = Vec::<F>::new();
    let mut sub_res = Vec::<F>::new();
    for i in 0..=n-1 {
        add_res.push(ran_vec1[i]+ran_vec2[i]);
        sub_res.push(ran_vec1[i]-ran_vec2[i]);
        };
    let t_add=vec_add_sub(ran_vec1.clone(), ran_vec2.clone(), true);
    let t_sub=vec_add_sub(ran_vec1.clone(), ran_vec2.clone(), false);
    assert_vector_eq(t_add.clone(),add_res.clone());
    println!("vector addition functions test passed\n");
    assert_vector_eq(t_sub.clone(),sub_res.clone());
    println!("vector function comparison test passed\n");

    //check if add_res+sub_res = 2 a_i , add_res-sub_res=2 b_i 
    for i in 0..=n-1 {
        assert_eq!(F::add(add_res[i],sub_res[i]),F::double(&ran_vec1[i]));
        assert_eq!(F::sub(add_res[i],sub_res[i]),F::double(&ran_vec2[i]));
        };          
    println!("naive vector addition test passed");
        //check if (add_res_i + sub_res)* (add_res_i-sub_res_i) = 4 a_i * b_i 
    for i in 0..=n-1 {
        let a1 = F::add(add_res[i],sub_res[i]);
        let a2 = F::sub(add_res[i],sub_res[i]);
        assert_eq!(F::mul(a1, a2),F::mul(F::double(&ran_vec1[i]),F::double(&ran_vec2[i]))); 
        };
    println!("vector multiplication test passed");
        let padded:Vec<F> = zero_pad(2*n, ran_vec);
        assert_eq!(2*n,padded.len());
        println!("zero pad test passed");
        let t1 = vec_add_sub(t_add.clone(), t_sub.clone(),true);
        let t2 = vec_add_sub(t_add, t_sub,false);
        assert_eq!(vec_hadamard_prod(t1, t2),mul_vec_by_scalar(vec_hadamard_prod(ran_vec1, ran_vec2),F::from(4)));
        println!("Vector identity test (a+b).(a-b) - 4 a.b =0 passed");
        //println!("{:?}", ran_vec);
        //println!("{:?}", padded);   
}

// checks if a.I = a 
//note that F::one() is in some non trivial representation.
#[test]
pub fn mul_identity_test() {
    let mut rng = test_rng();
    let one= F::one();
    let test = F::rand(& mut rng);
    let prod = one.mul(&test);
    assert_eq!(prod,test);
}