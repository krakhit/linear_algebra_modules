use std::{clone, string, vec, iter::Skip};
use ark_std::{rand::{rngs::StdRng,RngCore},test_rng,ops::{Mul,Add,Sub,Div}, UniformRand};
use ark_ff::{Field,PrimeField,FpParameters, SquareRootField, bytes,Zero,One,Fp256, BigInteger, BigInteger256,FftField};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
    UVPolynomial,evaluations, domain,
};
use ark_bls12_381::{Fr as F,FrParameters};
use linear_algebra_modules::vector_ops::*;

#[test]
/// Sanity checks for FFT/iFFT
pub fn fft_sanity_check() {
  let mut rng = test_rng();
  let init_domain_size = 256;
  let test_vec:Vec<F> = gen_rand_vec(256, &mut rng);
  let init_domain= GeneralEvaluationDomain::<F>::new(init_domain_size).unwrap();
 
  //go to poly eval form
  let test_coeffs = GeneralEvaluationDomain::ifft(&init_domain,&test_vec);
  //go back to coeff form
  let test_evals=GeneralEvaluationDomain::fft(&init_domain,&test_coeffs);
  // check if a = fft_n(ifft_n(a))
  assert_vector_eq(test_evals,test_vec);
  println!("FFT sanity checks passed.")
}

#[test]
    //given vector can be considered as polynomial in eval form, the purpose
    //is to check the .inverse().wrap() function
pub fn poly_division_test(){
    let mut rng = test_rng();
    let mut a= Vec::<F>::new();
    let mut b= Vec::<F>::new();
    let mut c= Vec::<F>::new();
    let init_domain_size = 256;
    for i in 0..=init_domain_size-1{
        a.push(F::rand(&mut rng));
        b.push(a[i].inverse().unwrap());
        c.push(F::one());
    }
    //sanity
    let resL = vec_hadamard_prod(a.clone(), b.clone());
    assert_vector_eq(resL, c.clone());  
    //inversion function check
    let resR=poly_division_eval_form(a.clone(), a.clone());
    assert_vector_eq(resR, c);
}


#[test]
pub fn fft_degree_sanity_check(){
    let mut rng = test_rng();
    
    let mut a= Vec::<F>::new();
    let mut b= Vec::<F>::new();
    let mut c= Vec::<F>::new();
    let mut zh = Vec::<F>::new();
    let init_domain_size = 8;
    for i in 0..=init_domain_size-1{
        a.push(F::rand(&mut rng));
        b.push(F::rand(&mut rng));
        c.push(F::mul(a[i],b[i]));
        if i == 0 {
            zh.push(F::zero()-F::one());
        }  else {
            zh.push(F::zero())
        }
    }
    zh.push(F::one());
    let init_domain= GeneralEvaluationDomain::<F>::new(init_domain_size).unwrap();

    //checks that this indeed is a vanishing poly (true upto a struct)
    //println!("{:?}",zh.clone());
    //println!("{:?}",GeneralEvaluationDomain::vanishing_polynomial(&init_domain));
        
    // initial conversion to coefficient form
    let mut a_poly1 = GeneralEvaluationDomain::ifft(&init_domain,&a);
    let mut b_poly1= GeneralEvaluationDomain::ifft(&init_domain,&b);
    let mut c_poly1 = GeneralEvaluationDomain::ifft(&init_domain,&c);
    
    //Method 1 h =(CosetFFT(a).CosetFFT(b) - CosetFFT(c))/zh_eval_at_coset
    //then compute InvCosetFFT(h) gives coeff of h
    GeneralEvaluationDomain::coset_fft_in_place(&init_domain, &mut a_poly1);
    GeneralEvaluationDomain::coset_fft_in_place(&init_domain, &mut b_poly1);
    GeneralEvaluationDomain::coset_fft_in_place(&init_domain, &mut c_poly1);
    let mut nr1 = vec_add_sub(vec_hadamard_prod(a_poly1, b_poly1),c_poly1,false);
    GeneralEvaluationDomain::divide_by_vanishing_poly_on_coset_in_place(&init_domain, &mut nr1);
    GeneralEvaluationDomain::coset_ifft_in_place(&init_domain, &mut nr1);
    
    // (a.b) is degree 2n, and c is degree n, while the coset does not care about this or does it?   
    //Method 2 :stupid test
    //t1= (CosetFFT(a).CosetFFT(b) /zh_eval_coset 
    //t2 = (cosetFFT(c))/zh_eval_coset
    //h = invCosetFFT(t1-t2) is the check
    // this will pass if every thing is evaluated in the same coset. 

    let mut a_poly2 = GeneralEvaluationDomain::ifft(&init_domain,&a);
    let mut b_poly2 = GeneralEvaluationDomain::ifft(&init_domain,&b);
    let mut c_poly2 = GeneralEvaluationDomain::ifft(&init_domain,&c);

    GeneralEvaluationDomain::coset_fft_in_place(&init_domain, &mut a_poly2);
    GeneralEvaluationDomain::coset_fft_in_place(&init_domain, &mut b_poly2);
    let mut t1= vec_hadamard_prod(a_poly2, b_poly2);
    GeneralEvaluationDomain::divide_by_vanishing_poly_on_coset_in_place(&init_domain, &mut t1);
    GeneralEvaluationDomain::coset_fft_in_place(&init_domain, &mut c_poly2);
    GeneralEvaluationDomain::divide_by_vanishing_poly_on_coset_in_place(&init_domain, &mut c_poly2);
    let mut nr2= vec_add_sub(t1, c_poly2, false);
    GeneralEvaluationDomain::coset_ifft_in_place(&init_domain, &mut nr2);

    //it is not a surprise that this stupid check passes
    assert_vector_eq(nr1, nr2);
    // it is just arkworks sanity checks.. no big deal    
}

#[test]
/// This test checks that given a polynomial F(X) = \sum_i a_i X^i
/// the relation  CosetFFT_g (F(X)) = FFT (F(gX)) holds
/// where CosetFFT_g is a DFT with the twiddle factors (g, gw, gw^2,...)
/// regular FFT is a DFT with the twiddle factors (1,w,w^2...)
pub fn interpolate_on_coset_test(){
    let mut rng = test_rng();
    let mut a= Vec::<F>::new();
     let init_domain_size = 8;
    for i in 0..=init_domain_size-1{
        a.push(F::rand(&mut rng));
    }
    let init_domain= GeneralEvaluationDomain::<F>::new(init_domain_size).unwrap();
    //F(X) =\sum_i a_i X^i
    let mut a_poly1 = GeneralEvaluationDomain::ifft(&init_domain,&a);
    //copy for future use
    let mut a_poly2  = a_poly1.clone();
    // CosetFFT_g (F(X)) with domain (g, gw,gw^2,....) by default algo modifying in place
    GeneralEvaluationDomain::coset_fft_in_place(&init_domain, &mut a_poly1);
    // clone CosetFFT_g (F(X))
    let mut a_poly_1_coset_g_eval = a_poly1.clone();
    // g=mu for coset used by arkworks
    let mu = F::multiplicative_generator();
    // gen (1,g,g^2....)
    let powers_of_mu = gen_pow_tau(8,mu);
    // given F(X) \sum a_i X^i below compute the coeff vector (a_0, a_1 g, a_2 g^2...) that represents F(gX)
    let mut a_poly1_inter= vec_hadamard_prod(powers_of_mu,a_poly2.clone());
    //clone it for future use
    let mut a_eval_at_gX=a_poly1_inter.clone();
    // do ordinary FFT(F(gX))
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut a_poly1_inter);
    
    // This proves the relation CosetFFT_g (F(X)) = FFT (F(gX)) compare evals with evals
    assert_vector_eq(a_poly_1_coset_g_eval.clone(), a_poly1_inter.clone());

    //inverse relation FFT(CosetFFT_g (F(X)) = f(X) = F(gX)
    // g^{-1}
    let mu_inv=mu.inverse().unwrap();
    // (1,g^{-1}, g^{-2},...)
    let pows_of_mu_inv = gen_pow_tau(8, mu_inv);
    //IFFT(CosetFFT_g(F(X))) =f(X)
    GeneralEvaluationDomain::ifft_in_place(&init_domain, &mut a_poly_1_coset_g_eval);
    // f(g^{-1}X)
    let mut a_poly_1_final= vec_hadamard_prod(pows_of_mu_inv, a_poly_1_coset_g_eval.clone());
    //checks F(X) = f(g^{-1}X)
    assert_vector_eq(a_poly2.clone(), a_poly_1_final.clone());
    //now use coset this way
}

#[test]

pub fn coset_splice_test_deg_2n(){
    let mut rng = test_rng();
    let mut a= Vec::<F>::new();
    let mut b= Vec::<F>::new();
     let init_domain_size = 8;
    for i in 0..=init_domain_size-1{
        a.push(F::rand(&mut rng));
        b.push(F::rand(&mut rng));
    }
    //define domains
    let init_domain= GeneralEvaluationDomain::<F>::new(init_domain_size).unwrap();
    let prod_domain = GeneralEvaluationDomain::<F>::new(2*init_domain_size).unwrap();
    // initial conversion to coefficient form
    let mut a_poly1 = GeneralEvaluationDomain::ifft(&init_domain,&a);
    let mut b_poly1= GeneralEvaluationDomain::ifft(&init_domain,&b);
    
    //----Start Method 1 standard convolution ------
    let mut a_poly1_std = a_poly1.clone();
    let mut b_poly1_std = b_poly1.clone();
    //compute 2n evals
    GeneralEvaluationDomain::fft_in_place(&prod_domain, & mut a_poly1_std);
    GeneralEvaluationDomain::fft_in_place(&prod_domain, & mut b_poly1_std);
    let mut std_convolution = vec_hadamard_prod(a_poly1_std, b_poly1_std);
    //result of convolution with standard method = std_convolution
    GeneralEvaluationDomain::ifft_in_place(&prod_domain, & mut std_convolution);
    //----End Method 1 standard convolution ------

    
    //-- start coset method --//
    //define coset generator
    let mu = F::multiplicative_generator(); 
    //NOTE:vone can just set mu to F::one() if we dont need to divide by vanishing poly later

    //define extended omega factor : it appears here to compute parts of large FFT as small coset slices
    // FFT(f(zeta * extended_omega_factor * X), n), is the mth part in a mn length vector
    // where `extended_omega_factor` is `extended_omega^i` with `i` in `[0, m)` 
    // F(x) = a_0 + a_1 X+ a_2 X^2 .... F(mu X) = a_0 + a_1 mu X + a_2 mu^2 X
    //here we are going to use cosetFFT_0(F(X)) =FFT(F(mu X)), where the domain of cosetFFT_0 is mu*(1, w_n, w_n^2....)
    //and  CosetFFT_1(F(X)) = FFT(F(mu omega_{2n} X)) where domain of coset FFT_1 is mu*w_{2n}*(1,  w_n, w_n^2,...) 
    // this is gen of w_{2n}
    let ext_om_factor = prod_domain.element(1);
    
    //gen pows of mu to compute coset_zero
    let coset_zero_part = gen_pow_tau(8,mu);    
    //gen pows of mu*om_{2n} to compute coset_one
    let coset_one_part = gen_pow_tau(8,mu.mul(ext_om_factor));
    //gen pows of mu^{-1} for final IFFT 
    let coset_inv_full = gen_pow_tau(16, mu.inverse().unwrap());

    //copy original poly in coeff form//
    let mut a_poly2 = a_poly1.clone();
    let mut b_poly2 = b_poly1.clone();
    //compute coset zero evals : gen n evals of F(mu X)
    let mut a_poly_coset_zero_evals = vec_hadamard_prod(coset_zero_part.clone(), a_poly2.clone());
    let mut b_poly_coset_zero_evals = vec_hadamard_prod(coset_zero_part.clone(), b_poly2.clone());
    //compute FFT(F(mu X)) = CosetFFT_0(F(X))
    GeneralEvaluationDomain::fft_in_place(&init_domain,&mut a_poly_coset_zero_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain,&mut b_poly_coset_zero_evals);

    // we are using the tric FFT(F(gx)) = CosetFFT_0(F(x))
    //---compute F( mu * w_{2n} X)
    let mut a_poly_coset_mu_evals = vec_hadamard_prod(coset_one_part.clone(), a_poly2);  
    let mut b_poly_coset_mu_evals = vec_hadamard_prod(coset_one_part.clone(), b_poly2);     
    //---compute FFT(F( mu * w_{2n} X)) = CosetFFT_1(F(X))
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut a_poly_coset_mu_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut b_poly_coset_mu_evals);
    
    //due to eval domain element wise product (A.B)
    let coset_convolution_in_zero = vec_hadamard_prod(a_poly_coset_zero_evals, b_poly_coset_zero_evals); 
    let coset_convolution_in_mu = vec_hadamard_prod(a_poly_coset_mu_evals, b_poly_coset_mu_evals); 
    // coset 0 F(mu), F(mu w_n), F(mu w_n^2)
    //coset 1 F(mu w_{2n} )F(mu w_{2n}^2)
    //splicing the vector in the order (F(mu) , F(mu w_{2n}), F (mu w_n) , F(mu w_{2n} w_n),.....)
    //for mu=1 it will look like F(1), F(sqrt{w}), F(w), F(w^{3/2})....
    let mut coset_convolution_spliced = Vec::<F>::new();
    for i in 0..=init_domain_size-1 {
        coset_convolution_spliced.push(coset_convolution_in_zero[i]);
        coset_convolution_spliced.push(coset_convolution_in_mu[i]);
    }
    //dp the IFFT(F(mu^{-1} X)) = IcosetFFT_0(F(X))
    GeneralEvaluationDomain::ifft_in_place(&prod_domain, &mut coset_convolution_spliced);
    //get rid of the mus i.e mu^{-1}
    let final_coset = vec_hadamard_prod(coset_convolution_spliced,coset_inv_full); 
    assert_vector_eq(final_coset, std_convolution);
}

#[test]

pub fn coset_splice_test_deg_3n(){
    let mut rng = test_rng();
    let mut a= Vec::<F>::new();
    let mut b= Vec::<F>::new();
    let mut c= Vec::<F>::new();
     let init_domain_size = 8;
    for i in 0..=init_domain_size-1{
        a.push(F::rand(&mut rng));
        b.push(F::rand(&mut rng));
        c.push(F::rand(&mut rng));
    }
    //define domains
    let init_domain= GeneralEvaluationDomain::<F>::new(init_domain_size).unwrap();
    let prod_domain = GeneralEvaluationDomain::<F>::new(4*init_domain_size).unwrap();
    // initial conversion to coefficient form
    let mut a_poly1 = GeneralEvaluationDomain::ifft(&init_domain,&a);
    let mut b_poly1= GeneralEvaluationDomain::ifft(&init_domain,&b);
    let mut c_poly1= GeneralEvaluationDomain::ifft(&init_domain,&c);
    
    //----Start Method 1 standard convolution ------
    let mut a_poly1_std = a_poly1.clone();
    let mut b_poly1_std = b_poly1.clone();
    let mut c_poly1_std = c_poly1.clone();
    //compute 3n evals
    GeneralEvaluationDomain::fft_in_place(&prod_domain, & mut a_poly1_std);
    GeneralEvaluationDomain::fft_in_place(&prod_domain, & mut b_poly1_std);
    GeneralEvaluationDomain::fft_in_place(&prod_domain, & mut c_poly1_std);

    let mut temp = vec_hadamard_prod(a_poly1_std, b_poly1_std);
    let mut std_convolution = vec_hadamard_prod(c_poly1_std, temp);
    //result of convolution with standard method = std_convolution
    GeneralEvaluationDomain::ifft_in_place(&prod_domain, & mut std_convolution);
    //----End Method 1 standard convolution ------

    
    //-- start coset method --//
    //define coset generator
//    let mu = F::multiplicative_generator();
let mu = F::one();
    //NOTE:vone can just set mu to F::one() if we dont need to divide by vanishing poly later

    //define extended omega factor : it appears here to compute parts of large FFT as small coset slices
    // FFT(f(zeta * extended_omega_factor * X), n), is the mth part in a mn length vector
    // where `extended_omega_factor` is `extended_omega^i` with `i` in `[0, m)` 

    //here we are going to use cosetFFT_0(F(X)) =FFT(F(mu X)), where the domain of cosetFFT_0 is mu*(1, w_n, w_n^2....)
    //and  CosetFFT_1(F(X)) = FFT(F(mu omega_{3n} X)) where domain of coset FFT_1 is mu*w_{3n}*(1,  w_n, w_n^2,...)
     //and  CosetFFT_2(F(X)) = FFT(F(mu omega_{3n}^2 X)) where domain of coset FFT_1 is mu*w_{3n}^2*(1,  w_n, w_n^2,...)
    
    // this is gen of w_{3n} (extended domain)
    let ext_om_factor = prod_domain.element(1);
    
    //gen pows of mu to compute coset_zero
    let coset_zero_part = gen_pow_tau(8,mu);    
    //gen pows of mu*om_{3n} to compute coset_one
    let coset_one_part = gen_pow_tau(8,mu.mul(ext_om_factor));
    let coset_two_part = gen_pow_tau(8,mu.mul(ext_om_factor.square()));
    let coset_three_part = gen_pow_tau(8,mu.mul(ext_om_factor.square()).mul(ext_om_factor));
    //gen pows of mu^{-1} for final IFFT 
    let coset_inv_full = gen_pow_tau(32, mu.inverse().unwrap());

    //copy original poly in coeff form//
    let mut a_poly2 = a_poly1.clone();
    let mut b_poly2 = b_poly1.clone();
    let mut c_poly2 = c_poly1.clone();
    //compute coset zero evals : gen n evals of F(mu X)
    let mut a_poly_coset_zero_evals = vec_hadamard_prod(coset_zero_part.clone(), a_poly2.clone());
    let mut b_poly_coset_zero_evals = vec_hadamard_prod(coset_zero_part.clone(), b_poly2.clone());
    let mut c_poly_coset_zero_evals = vec_hadamard_prod(coset_zero_part.clone(), c_poly2.clone());
    //compute FFT(F(mu X)) = CosetFFT_0(F(X))
    GeneralEvaluationDomain::fft_in_place(&init_domain,&mut a_poly_coset_zero_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain,&mut b_poly_coset_zero_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain,&mut c_poly_coset_zero_evals);
    

    // we are using the tric FFT(F(gx)) = CosetFFT_0(F(x))
    //---compute F( mu * w_{3n} X)
    let mut a_poly_coset_mu_evals = vec_hadamard_prod(coset_one_part.clone(), a_poly2.clone());  
    let mut b_poly_coset_mu_evals = vec_hadamard_prod(coset_one_part.clone(), b_poly2.clone());
    let mut c_poly_coset_mu_evals = vec_hadamard_prod(coset_one_part.clone(), c_poly2.clone());          
    //---compute FFT(F( mu * w_{2n} X)) = CosetFFT_1(F(X))
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut a_poly_coset_mu_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut b_poly_coset_mu_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut c_poly_coset_mu_evals);
    
    //---compute F( mu * w_{3n}^2 X)
    let mut a_poly_coset_mu2_evals = vec_hadamard_prod(coset_two_part.clone(), a_poly2.clone());  
    let mut b_poly_coset_mu2_evals = vec_hadamard_prod(coset_two_part.clone(), b_poly2.clone());
    let mut c_poly_coset_mu2_evals = vec_hadamard_prod(coset_two_part.clone(), c_poly2.clone());          
    //---compute FFT(F( mu * w_{2n}^2 X)) = CosetFFT_2(F(X))
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut a_poly_coset_mu2_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut b_poly_coset_mu2_evals);
    GeneralEvaluationDomain::fft_in_place(&init_domain, &mut c_poly_coset_mu2_evals);

   //---compute F( mu * w_{3n}^3 X)
   let mut a_poly_coset_mu3_evals = vec_hadamard_prod(coset_three_part.clone(), a_poly2.clone());  
   let mut b_poly_coset_mu3_evals = vec_hadamard_prod(coset_three_part.clone(), b_poly2.clone());
   let mut c_poly_coset_mu3_evals = vec_hadamard_prod(coset_three_part.clone(), c_poly2.clone());          
   //---compute FFT(F( mu * w_{2n}^3 X)) = CosetFFT_3(F(X))
   GeneralEvaluationDomain::fft_in_place(&init_domain, &mut a_poly_coset_mu3_evals);
   GeneralEvaluationDomain::fft_in_place(&init_domain, &mut b_poly_coset_mu3_evals);
   GeneralEvaluationDomain::fft_in_place(&init_domain, &mut c_poly_coset_mu3_evals);   
    
    //due to eval domain element wise product (A.B.C)
    let temp10 = vec_hadamard_prod(a_poly_coset_zero_evals, b_poly_coset_zero_evals); 
    let coset_convolution_in_zero = vec_hadamard_prod(c_poly_coset_zero_evals, temp10); 

    let temp11= vec_hadamard_prod(a_poly_coset_mu_evals, b_poly_coset_mu_evals); 
    let coset_convolution_in_mu = vec_hadamard_prod(c_poly_coset_mu_evals, temp11); 

    let temp12= vec_hadamard_prod(a_poly_coset_mu2_evals, b_poly_coset_mu2_evals); 
    let coset_convolution_in_mu2 = vec_hadamard_prod(c_poly_coset_mu2_evals, temp12); 

    let temp13= vec_hadamard_prod(a_poly_coset_mu3_evals, b_poly_coset_mu3_evals); 
    let coset_convolution_in_mu3 = vec_hadamard_prod(c_poly_coset_mu3_evals, temp13); 

    //splicing the vector in the order (F(mu) , F(mu w_{2n}), F (mu w_n) , F(mu w_{2n} w_n),.....)
    //for mu=1 it will look like F(1), F(sqrt{w}), F(w), F(w^{3/2})....
    let mut coset_convolution_spliced = Vec::<F>::new();
    for i in 0..=init_domain_size-1 {
        coset_convolution_spliced.push(coset_convolution_in_zero[i]);
        coset_convolution_spliced.push(coset_convolution_in_mu[i]);
        coset_convolution_spliced.push(coset_convolution_in_mu2[i]);
        coset_convolution_spliced.push(coset_convolution_in_mu3[i]);
    }//
    // F(x) = a_0 + a_2 X^2 + X(a_1 + a_3 X^2)
    //dp the IFFT(F(mu^{-1} X)) = IcosetFFT_0(F(X))
    GeneralEvaluationDomain::ifft_in_place(&prod_domain, &mut coset_convolution_spliced);
    //get rid of the mus i.e mu^{-1}
    let final_coset_convolution = vec_hadamard_prod(coset_convolution_spliced,coset_inv_full); 
    assert_vector_eq(final_coset_convolution, std_convolution);
}
