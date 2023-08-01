pub mod vector_ops {
    use ark_ff::fields::Field;
    use ark_std::{rand::RngCore};
    use std::{iter, vec};
    
    /// generates a random vector of a given size
    pub fn gen_rand_vec<F:Field,R:RngCore>(
        max_vec_size: usize,
        rng: &mut R,
    ) -> Vec<F> {
        let mut ran_vec= Vec::<F>::new();
        for _i in 0..max_vec_size {
            ran_vec.push(F::rand(rng));
            };
        ran_vec
    }
    ///generate a random vector of the form $\{1,\beta,\beta^2,\ldots\\, \beta^{n-1}}$, where 
    /// n is the maximum vector size given by the user
    pub fn gen_pow_tau<F:Field>(
        max_vec_size: usize,
        tau: F,
    ) -> Vec<F> {
        //equivalent of secret scalar
        let gen = F::one();
        let powers_of_tau: Vec<F> = iter::successors(Some(gen), |p| Some(*p * tau))
            .take(max_vec_size)
            .collect();
        //outputs the powers of tau
        powers_of_tau
    }
    /// given a input vector, zero pads the input vector to a given size. 
    /// if the size of the input vector is less than the zero pad target size, then
    /// it outputs the input vector
    pub fn zero_pad<F:Field>(
        target_zero_pad_size: usize,
        input_vec: Vec<F>,
    ) -> Vec<F> {
        let mut padded_vec = input_vec;
        let l = padded_vec.len();
        let to_pad:usize = if target_zero_pad_size>l{
            target_zero_pad_size-l
        } else{
            0
        };
        for _i in 0..to_pad {
            padded_vec.push(F::zero());
            };
        padded_vec
    }
    /// adds or subtracts a vector, in this definition, bool=true does addition
    /// and bool =false does subtraction. If vectors are of different lengths, panics
    pub fn vec_add_sub<F:Field>(
        vec1: Vec<F>,
        vec2: Vec<F>,
        arg: bool,
    )-> Vec<F> {
        assert_eq!(vec1.len(),vec2.len(),"Vectors must be of same length!");
        let mut result_vec:Vec<F> = Vec::<F>::new();
        if arg {
            for _i in 0.. vec1.len() {
                result_vec.push(vec1[_i]+vec2[_i]);
            }
        } else {
            for _i in 0..vec1.len() {
                result_vec.push(vec1[_i]-vec2[_i]);
            }
        }
        result_vec
    }
    ///Computes hadamard product of two vectors of the same length, panics if vectors are of different length
    pub fn vec_hadamard_prod<F:Field>(
        vec1: Vec<F>,
        vec2: Vec<F>,
    )-> Vec<F> {
        assert_eq!(vec1.len(),vec2.len(),"Vectors must be of same length!");

        let mut result_vec:Vec<F> = Vec::<F>::new();
        for _i in 0.. vec1.len() {
                result_vec.push(vec1[_i]*vec2[_i]);
        };
        result_vec
    }

    /// computes scalar product of two vectors, panics if vectors are of different length
    pub fn vec_inner_prod<F:Field>(
        vec1: Vec<F>,
        vec2: Vec<F>,
    )-> F {
        assert_eq!(vec1.len(),vec2.len(),"Vectors must be of same length!");

        let mut result = F::zero();
        for _i in 0.. vec1.len() {
                result += vec1[_i]*vec2[_i];
        };
        result
    }

    ///asserts vector equality, panics if vectors are of different length
    pub fn assert_vector_eq<F:Field>(
        vec1: Vec<F>,
        vec2: Vec<F>,
    ) {
        assert_eq!(vec1.len(),vec2.len(),"Vectors must be of same length!");
        for _i in 0.. vec1.len() {
                assert_eq!(vec1[_i],vec2[_i],"vectors are not equal!")
        };
    }

     ///Computes sum of all elements in a vector
     pub fn vector_trace<F:Field>(
        vec1: Vec<F>,
    )-> F {
        let mut result = F::zero();
        for _i in 0.. vec1.len() {
                result += vec1[_i];
        };
        result
    }

    pub fn mul_vec_by_scalar <F:Field> (
        vec1: Vec<F>,
        scalar: F,
    ) -> Vec<F> {
        let mut result_vec:Vec<F> = Vec::<F>::new();
        for _i in 0..vec1.len() {
            result_vec.push(vec1[_i]*scalar);
        }
        result_vec
    }
    
    /// Given two polynomials vec1 and vec2 in evaluation form, does 
    ///element wise division. vec1[i]/vec2[i]
    pub fn poly_division_eval_form <F:Field> (
        vec1: Vec<F>,
        vec2: Vec<F>,
    ) -> Vec<F> {
        let mut result_vec:Vec<F> = Vec::<F>::new();
        for _i in 0..vec1.len() {
            result_vec.push(vec1[_i] * ((vec2[_i].inverse().unwrap())));
        }
        result_vec
    }
    /// evaluate a polynomial in coefficient form! at a given point and give the result
    pub fn eval_poly <F:Field> (
        vec: Vec<F>,
        tau: F,
    ) -> F {
        let dom = vec.len();
        let powers:Vec<F> = gen_pow_tau(dom,tau);
        let evals = vec_inner_prod(vec, powers);
        evals
    }

    ///create upper triangular toeplitz with 1, beta, beta^2...
    pub fn upper_triangular_toeplitz <F:Field> (
        rows: usize,
        beta: F,
    ) -> Vec<Vec<F>> {
        let vec_size = &rows;
        let powers:Vec<F> = gen_pow_tau(rows, beta);
       
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
        ToeplitzM
      }

      pub fn matrix_vec_multiply<F:Field> (
        matrix: Vec<Vec<F>>,
        vector: Vec<F>
      )-> Vec<F>{
        assert_eq!(&matrix.len(),&vector.len());
        let lr = matrix.len();
        let mut temp= Vec::<F>::new();
        for i in 0..=lr-1 {
            temp.push(vec_inner_prod(matrix[i].clone(), vector.clone()));
        }
        temp 
      }
    }